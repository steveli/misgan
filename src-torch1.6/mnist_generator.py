import torch
import torch.nn as nn
import torch.nn.functional as F


def add_data_transformer(self):
    self.transform = lambda x: torch.sigmoid(x).view(-1, 1, 28, 28)


def add_mask_transformer(self, temperature=.66, hard_sigmoid=(-.1, 1.1)):
    """
    hard_sigmoid:
        False:  use sigmoid only
        True:   hard thresholding
        (a, b): hard thresholding on rescaled sigmoid
    """
    self.temperature = temperature
    self.hard_sigmoid = hard_sigmoid

    view = -1, 1, 28, 28
    if hard_sigmoid is False:
        self.transform = lambda x: torch.sigmoid(x / temperature).view(*view)
    elif hard_sigmoid is True:
        self.transform = lambda x: F.hardtanh(
            x / temperature, 0, 1).view(*view)
    else:
        a, b = hard_sigmoid
        self.transform = lambda x: F.hardtanh(
            torch.sigmoid(x / temperature) * (b - a) + a, 0, 1).view(*view)


# Must sub-class ConvGenerator to provide transform()
class ConvGenerator(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.DIM = 64
        self.latent_size = latent_size

        self.preprocess = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 4 * self.DIM),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 5),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 5),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(self.DIM, 1, 8, stride=2)

    def forward(self, input):
        net = self.preprocess(input)
        net = net.view(-1, 4 * self.DIM, 4, 4)
        net = self.block1(net)
        net = net[:, :, :7, :7]
        net = self.block2(net)
        net = self.deconv_out(net)
        return self.transform(net)


# Must sub-class FCGenerator to provide transform()
class FCGenerator(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()
        self.latent_size = latent_size
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
        )

    def forward(self, input):
        net = self.fc(input)
        return self.transform(net)


class ConvDataGenerator(ConvGenerator):
    def __init__(self, latent_size=128):
        super().__init__(latent_size=latent_size)
        add_data_transformer(self)


class FCDataGenerator(FCGenerator):
    def __init__(self, latent_size=128):
        super().__init__(latent_size=latent_size)
        add_data_transformer(self)


class ConvMaskGenerator(ConvGenerator):
    def __init__(self, latent_size=128, temperature=.66,
                 hard_sigmoid=(-.1, 1.1)):
        super().__init__(latent_size=latent_size)
        add_mask_transformer(self, temperature, hard_sigmoid)


class FCMaskGenerator(FCGenerator):
    def __init__(self, latent_size=128, temperature=.66,
                 hard_sigmoid=(-.1, 1.1)):
        super().__init__(latent_size=latent_size)
        add_mask_transformer(self, temperature, hard_sigmoid)
