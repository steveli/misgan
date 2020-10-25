import torch
import torch.nn as nn
import torch.nn.functional as F


# Must sub-class Imputer to provide fc1
class Imputer(nn.Module):
    def __init__(self, arch=(784, 784)):
        super().__init__()
        # self.fc1 = nn.Linear(784, arch[0])
        self.fc2 = nn.Linear(arch[0], arch[1])
        self.fc3 = nn.Linear(arch[1], arch[0])
        self.fc4 = nn.Linear(arch[0], 784)
        self.transform = lambda x: torch.sigmoid(x).view(-1, 1, 28, 28)

    def forward(self, input, data, mask):
        net = input.view(input.size(0), -1)
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        net = F.relu(self.fc3(net))
        net = self.fc4(net)
        net = self.transform(net)
        # return data * mask + net * (1 - mask)
        # NOT replacing observed part with input data for computing
        # autoencoding loss.
        return net


class ComplementImputer(Imputer):
    def __init__(self, arch=(784, 784)):
        super().__init__(arch=arch)
        self.fc1 = nn.Linear(784, arch[0])

    def forward(self, input, mask, noise):
        net = input * mask + noise * (1 - mask)
        return super().forward(net, input, mask)


class MaskImputer(Imputer):
    def __init__(self, arch=(784, 784)):
        super().__init__(arch=arch)
        self.fc1 = nn.Linear(784 * 2, arch[0])

    def forward(self, input, mask, noise):
        batch_size = input.size(0)
        net = torch.cat(
            [(input * mask + noise * (1 - mask)).view(batch_size, -1),
             mask.view(batch_size, -1)], 1)
        return super().forward(net, input, mask)


class FixedNoiseDimImputer(Imputer):
    def __init__(self, arch=(784, 784)):
        super().__init__(arch=arch)
        self.fc1 = nn.Linear(784 * 3, arch[0])

    def forward(self, input, mask, noise):
        batch_size = input.size(0)
        net = torch.cat([(input * mask).view(batch_size, -1),
                         mask.view(batch_size, -1),
                         noise.view(batch_size, -1)], 1)
        return super().forward(net, input, mask)
