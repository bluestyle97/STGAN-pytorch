import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import *
from models.stgan import Generator, Discriminator
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class STGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers, self.config.shortcut_layers, use_stu=self.config.use_stu, one_more_conv=self.config.one_more_conv)
        self.D = Discriminator(self.config.image_size, len(self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size)

        self.current_iteration = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir)

    def save_checkpoint(self):
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
        }
        D_state  = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
        }
        G_filename = 'G_{}.pth.tar'.format(self.current_iteration)
        D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
        torch.save(G_state, os.path.join(self.config.checkpoint_dir, G_filename))
        torch.save(D_state, os.path.join(self.config.checkpoint_dir, D_filename))

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            self.G.to(self.device)
            self.D.to(self.device)
            return
        G_filename = 'G_{}.pth.tar'.format(self.config.checkpoint)
        D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
        G_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, G_filename))
        D_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, D_filename))
        G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
        D_to_load = {k.replace('module.', ''): v for k, v in D_checkpoint['state_dict'].items()}
        self.current_iteration = self.config.checkpoint
        self.G.load_state_dict(G_to_load)
        self.D.load_state_dict(D_to_load)
        self.G.to(self.device)
        self.D.to(self.device)
        if self.config.mode == 'train':
            self.optimizer_G.load_state_dict(G_checkpoint['optimizer'])
            self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # get hair color indices
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(len(selected_attrs)):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # set one hair color to 1 and the rest to 0
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # reverse attribute value

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def run(self):
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'train':
                self.train()
            else:
                self.test()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
        finally:
            self.finalize()


    def train(self):
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)

        self.load_checkpoint()
        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(self.G, device_ids=list(range(self.config.ngpu)))
            self.D = nn.DataParallel(self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)
        x_sample, c_org_sample = next(val_iter)
        x_sample = x_sample.to(self.device)
        c_sample_list = self.create_labels(c_org_sample, self.config.attrs)
        c_sample_list.insert(0, c_org_sample)  # reconstruction

        self.g_lr = self.lr_scheduler_G.get_lr()[0]
        self.d_lr = self.lr_scheduler_D.get_lr()[0]

        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        for i in range(self.current_iteration, self.config.max_iters):
            self.G.train()
            self.D.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # fetch real images and labels
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader.train_loader)
                x_real, label_org = next(data_iter)

            # generate target domain labels randomly
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)         # input images
            c_org = c_org.to(self.device)           # original domain labels
            c_trg = c_trg.to(self.device)           # target domain labels
            label_org = label_org.to(self.device)   # labels for computing classification loss
            label_trg = label_trg.to(self.device)   # labels for computing classification loss

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # compute loss with real images
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # compute loss with fake images
            attr_diff = c_trg - c_org
            attr_diff = attr_diff * torch.rand_like(attr_diff) * (2 * self.config.thres_int)
            x_fake = self.G(x_real, attr_diff)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # compute loss for gradient penalty
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # backward and optimize
            d_loss_adv = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
            d_loss = d_loss_adv + self.config.lambda1 * d_loss_cls
            self.optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = d_loss.item()
            scalars['D/loss_adv'] = d_loss_adv.item()
            scalars['D/loss_cls'] = d_loss_cls.item()
            scalars['D/loss_real'] = d_loss_real.item()
            scalars['D/loss_fake'] = d_loss_fake.item()
            scalars['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.config.n_critic == 0:
                # original-to-target domain
                x_fake = self.G(x_real, attr_diff)
                out_src, out_cls = self.D(x_fake)
                g_loss_adv = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # target-to-original domain
                x_reconst = self.G(x_real, c_org - c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # backward and optimize
                g_loss = g_loss_adv + self.config.lambda3 * g_loss_rec + self.config.lambda2 * g_loss_cls
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # summarize
                scalars['G/loss'] = g_loss.item()
                scalars['G/loss_adv'] = g_loss_adv.item()
                scalars['G/loss_cls'] = g_loss_cls.item()
                scalars['G/loss_rec'] = g_loss_rec.item()

            self.current_iteration += 1

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if self.current_iteration % self.config.summary_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print('Elapsed [{}], Iteration [{}/{}]'.format(et, self.current_iteration, self.config.max_iters))
                for tag, value in scalars.items():
                    self.writer.add_scalar(tag, value, self.current_iteration)

            if self.current_iteration % self.config.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    x_sample = x_sample.to(self.device)
                    x_fake_list = [x_sample]
                    for c_trg_sample in c_sample_list:
                        attr_diff = c_trg_sample.to(self.device) - c_org_sample.to(self.device)
                        attr_diff = attr_diff * self.config.thres_int
                        x_fake_list.append(self.G(x_sample, attr_diff.to(self.device)))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    self.writer.add_image('sample', make_grid(self.denorm(x_concat.data.cpu()), nrow=1),
                                          self.current_iteration)
                    save_image(self.denorm(x_concat.data.cpu()),
                               os.path.join(self.config.sample_dir, 'sample_{}.jpg'.format(self.current_iteration)),
                               nrow=1, padding=0)

            if self.current_iteration % self.config.checkpoint_step == 0:
                self.save_checkpoint()

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

    def test(self):
        self.load_checkpoint()
        self.G.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.config.attrs)

                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    attr_diff = c_trg - c_org
                    x_fake_list.append(self.G(x_real, attr_diff.to(self.device)))
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.config.result_dir, 'sample_{}.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
