import os
import math
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


def make_dataset(root, mode, selected_attrs):
    assert mode in ['train', 'val', 'test']
    lines = [line.rstrip() for line in open(os.path.join(root, 'anno', 'list_attr_celeba.txt'), 'r')]
    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    if mode == 'train':
        lines = lines[:-2000]       # train set contains 200599 images
    if mode == 'val':
        lines = lines[-2000:-1800]  # val set contains 200 images
    if mode == 'test':
        lines = lines[-1800:]       # test set contains 1800 images

    items = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')
        items.append([filename, label])
    return items


class CelebADataset(data.Dataset):
    def __init__(self, root, mode, selected_attrs, transform=None):
        self.items = make_dataset(root, mode, selected_attrs)
        self.root = root
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, 'image', filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)


class CelebADataLoader(object):
    def __init__(self, root, mode, selected_attrs, crop_size=None, image_size=128, batch_size=16):
        if mode not in ['train', 'test',]:
            return

        transform = []
        if crop_size is not None:
            transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if mode == 'train':
            val_transform = transforms.Compose(transform)       # make val loader before transform is inserted
            val_set = CelebADataset(root, 'val', selected_attrs, transform=val_transform)
            self.val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            transform.insert(0, transforms.RandomHorizontalFlip())
            train_transform = transforms.Compose(transform)
            train_set = CelebADataset(root, 'train', selected_attrs, transform=train_transform)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_transform = transforms.Compose(transform)
            test_set = CelebADataset(root, 'test', selected_attrs, transform=test_transform)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
