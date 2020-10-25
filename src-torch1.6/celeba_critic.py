import torch.nn as nn


def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))


class ConvCritic(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        dim = 64
        self.ls = nn.Sequential(
            nn.Conv2d(n_channels, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, input):
        net = self.ls(input)
        return net.view(-1)
