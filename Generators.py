import torch.nn as nn


class SimpleGenerator(nn.Module):

    def __init__(self, nfeatures, n_out, leak):
        super(SimpleGenerator, self).__init__()

        self.g1_neurons = 256
        self.g2_neurons = 512
        self.g3_neurons = 1024

        self.leak = leak

        self.hidden0 = nn.Sequential(
            nn.Linear(nfeatures, self.g1_neurons),
            nn.LeakyReLU(self.leak)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(self.g1_neurons, self.g2_neurons),
            nn.LeakyReLU(self.leak)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(self.g2_neurons, self.g3_neurons),
            nn.LeakyReLU(self.leak)
        )

        self.out = nn.Sequential(
            nn.Linear(self.g3_neurons, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x