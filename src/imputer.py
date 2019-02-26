import torch
import torch.nn as nn
from fcnet import FullyConnectedNet
from unet import UnetSkipConnectionBlock


# Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, layers=5,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        mid_layers = layers - 2
        fact = 2**mid_layers

        unet_block = UnetSkipConnectionBlock(
            ngf * fact, ngf * fact, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True)

        for _ in range(mid_layers):
            half_fact = fact // 2
            unet_block = UnetSkipConnectionBlock(
                ngf * half_fact, ngf * fact, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer)
            fact = half_fact

        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class Imputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = lambda x: torch.sigmoid(x)

    def forward(self, input, mask, noise):
        net = input * mask + noise * (1 - mask)
        net = self.imputer_net(net)
        net = self.transform(net)
        # NOT replacing observed part with input data for computing
        # autoencoding loss.
        # return input * mask + net * (1 - mask)
        return net


class UNetImputer(Imputer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.imputer_net = UNet(*args, **kwargs)


class FullyConnectedImputer(Imputer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.imputer_net = FullyConnectedNet(*args, **kwargs)
