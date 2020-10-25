import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, weights, output_shape=None):
        super().__init__()
        n_layers = len(weights) - 1

        layers = [nn.Linear(weights[0], weights[1])]
        for i in range(1, n_layers):
            layers.extend([nn.ReLU(), nn.Linear(weights[i], weights[i + 1])])

        self.model = nn.Sequential(*layers)
        self.output_shape = output_shape

    def forward(self, input):
        output = self.model(input.view(input.shape[0], -1))
        if self.output_shape is not None:
            output = output.view(self.output_shape)
        return output
