import torch
import torch.nn as nn
from torchsummary import summary


class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self, ).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(in_dim * 2 + n_attrs, out_dim, 4, 2, 1)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state


class Generator(nn.Module):
    def __init__(self, n_attrs, dim=64, n_layers=5, stu_kernel_size=3):
        super(Generator, self).__init__()
        self.n_attrs = n_attrs
        self.n_layers = n_layers

        self.encoder = nn.ModuleList()
        in_channels = 3
        for i in range(self.n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, dim * 2 ** i, 4, 2, 1),
                nn.BatchNorm2d(dim * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = dim * 2 ** i

        self.stu = nn.ModuleList()
        for i in reversed(range(self.n_layers - 1)):
            self.stu.append(ConvGRUCell(n_attrs, dim * 2**i, dim * 2**i, stu_kernel_size))

        self.decoder = nn.ModuleList()
        for i in reversed(range(self.n_layers)):
            if i == self.n_layers - 1:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(dim * 2**i + n_attrs, dim * 2**i, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2**i),
                    nn.ReLU(inplace=True)
                ))
            elif i == 0:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(dim * 3, 3, 4, 2, 1),
                    nn.Tanh()
                ))
            else:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(dim * 3 * 2**i, dim * 2**i, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2**i),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x, a):
        encoder_out = []
        x_enc = x
        for layer in self.encoder:
            x_enc = layer(x_enc)
            encoder_out.append(x_enc)

        stu_out = []
        state = encoder_out[-1]
        for i, layer in enumerate(self.stu):
            output, state = layer(encoder_out[-(i+2)], state, a)
            stu_out.append(output)

        x_dec = encoder_out[-1]
        n, _, h, w = x_dec.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        decoder_out = self.decoder[0](torch.cat([x_dec, attr], dim=1))
        for i in range(1, self.n_layers):
            x_dec = torch.cat([decoder_out, stu_out[i-1]], dim=1)
            decoder_out = self.decoder[i](x_dec)
        return decoder_out


class Discriminator(nn.Module):
    def __init__(self, image_size, n_attrs, dim=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, dim * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(dim * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = dim * 2**i
        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(dim * 2**(n_layers-1) * feature_size**2, fc_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(dim * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc_dim, n_attrs),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att


if __name__ == '__main__':
    gen = Generator(5)
    summary(gen, [(3, 384, 384), (5,)], device='cpu')

    dis = Discriminator(384, 5)
    summary(dis, (3, 384, 384), device='cpu')