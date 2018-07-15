import numpy as np
import torch.nn as nn

class SimpleDiscriminator(nn.Module):

    def __init__(self, nfeatures, n_out, leak, drop_out):
        super(SimpleDiscriminator, self).__init__()

        self.leak = leak
        self.do = drop_out

        self.d1_neurons = 256
        self.d2_neurons = 128
        self.d3_neurons = 64

        self.nfeatures = int(np.prod(nfeatures[1:])) # TODO - Hacky, can this be fixed?

        self.hidden0 = nn.Sequential(
            nn.Linear(self.nfeatures, self.d1_neurons),
            nn.LeakyReLU(self.leak),
            nn.Dropout(self.do)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(self.d1_neurons, self.d2_neurons),
            nn.LeakyReLU(self.leak),
            nn.Dropout(self.do)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(self.d2_neurons, self.d3_neurons),
            nn.LeakyReLU(self.leak),
            nn.Dropout(self.do)
        )
        self.out = nn.Sequential(
            nn.Linear(self.d3_neurons,n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
