import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 4),
            nn.LeakyReLU(),

            nn.Linear(4, 4),
            nn.LeakyReLU(),

            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.layers(x)

print(NeuralNetwork())