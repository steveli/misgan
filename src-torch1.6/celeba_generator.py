import torch
import torch.nn as nn
import torch.nn.functional as F


def add_mask_transformer(self, temperature=.66, hard_sigmoid=(-.1, 1.1)):
    """
    hard_sigmoid:
        False:  use sigmoid only
        True:   hard thresholding
        (a, b): hard thresholding on rescaled sigmoid
    """
    self.temperature = temperature
    self.hard_sigmoid = hard_sigmoid

    if hard_sigmoid is False:
        self.transform = lambda x: torch.sigmoid(x / temperature)
    elif hard_sigmoid is True:
        self.transform = lambda x: F.hardtanh(
            x / temperature, 0, 1)
    else:
        a, b = hard_sigmoid
        self.transform = lambda x: F.hardtanh(
            torch.sigmoid(x / temperature) * (b - a) + a, 0, 1)


def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                           padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())


# Must sub-class ConvGenerator to provide transform()
class ConvGenerator(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        dim = 64

        self.l1 = nn.Sequential(
            nn.Linear(latent_size, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
                               padding=2, output_padding=1))

    def forward(self, input):
        net = self.l1(input)
        net = net.view(net.shape[0], -1, 4, 4)
        net = self.l2_5(net)
        return self.transform(net)


class ConvDataGenerator(ConvGenerator):
    def __init__(self, latent_size=128):
        self.out_channels = 3
        super().__init__(latent_size=latent_size)
        self.transform = lambda x: torch.sigmoid(x)


class ConvMaskGenerator(ConvGenerator):
    def __init__(self, latent_size=128, temperature=.66,
                 hard_sigmoid=(-.1, 1.1)):
        self.out_channels = 1
        super().__init__(latent_size=latent_size)
        add_mask_transformer(self, temperature, hard_sigmoid)
