import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class MaskedMNIST(Dataset):
    def __init__(self, data_dir='mnist-data', image_size=28, random_seed=0):
        self.rnd = np.random.RandomState(random_seed)
        self.image_size = image_size
        if image_size == 28:
            self.data = datasets.MNIST(
                data_dir, train=True, download=True,
                transform=transforms.ToTensor())
        else:
            self.data = datasets.MNIST(
                data_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size), transforms.ToTensor()]))
        self.generate_masks()

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, self.mask[index], label, index

    def __len__(self):
        return len(self.data)

    def generate_masks(self):
        raise NotImplementedError


class BlockMaskedMNIST(MaskedMNIST):
    def __init__(self, block_len=None, *args, **kwargs):
        self.block_len = block_len
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        d0_len = d1_len = self.image_size
        d0_min_len = 7
        d0_max_len = d0_len - d0_min_len
        d1_min_len = 7
        d1_max_len = d1_len - d1_min_len

        n_masks = len(self)
        self.mask = [None] * n_masks
        self.mask_info = [None] * n_masks
        for i in range(n_masks):
            if self.block_len is None:
                d0_mask_len = self.rnd.randint(d0_min_len, d0_max_len)
                d1_mask_len = self.rnd.randint(d1_min_len, d1_max_len)
            else:
                d0_mask_len = d1_mask_len = self.block_len

            d0_start = self.rnd.randint(0, d0_len - d0_mask_len + 1)
            d1_start = self.rnd.randint(0, d1_len - d1_mask_len + 1)

            mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            mask[d0_start:(d0_start + d0_mask_len),
                 d1_start:(d1_start + d1_mask_len)] = 1
            self.mask[i] = mask
            self.mask_info[i] = d0_start, d1_start, d0_mask_len, d1_mask_len


class IndepMaskedMNIST(MaskedMNIST):
    def __init__(self, obs_prob=.2, obs_prob_high=None, *args, **kwargs):
        self.prob = obs_prob
        self.prob_high = obs_prob_high
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        imsize = self.image_size
        prob = self.prob
        prob_high = self.prob_high
        n_masks = len(self)
        self.mask = [None] * n_masks
        for i in range(n_masks):
            if prob_high is None:
                p = prob
            else:
                p = self.rnd.uniform(prob, prob_high)
            self.mask[i] = torch.ByteTensor(imsize, imsize).bernoulli_(p)
