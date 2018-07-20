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

class ConvolutionalDiscriminator(nn.Module):

    def __init__(self, input_vector_dims, output_vector_dims, leak, dropout=None):
        """

        :param input_vector_dims: tuple with dimensions ( C, H, W)
        :param output_vector_dims: tuple with desired output dimensions ( C, H, W) - e.g. (1,28,28) for MNIST
        :param leak: leak for leaky RELU
        """
        super(ConvolutionalDiscriminator, self).__init__()

        self.initial_dimensions = input_vector_dims
        self.output_vector_dims = output_vector_dims

        self.kdim = 3

        self.h0 = {'channels': 32, 'kdim': self.kdim, 'stride': 1, 'padding': 1}
        self.h1 = {'channels': 16, 'kdim': self.kdim, 'stride': 2, 'padding': 1}
        self.h2 = {'channels': 32, 'kdim': self.kdim, 'stride': 2, 'padding': 1}
        self.h3 = {'channels': 32, 'kdim': self.kdim, 'stride': 1, 'padding': 1}

        total_down_fac = np.prod([layer['stride'] for layer in (self.h0, self.h1)])#, self.h2, self.h3)])

        self.leak = leak

        self.hidden0 = nn.Sequential(
            nn.Conv2d(self.initial_dimensions[0], self.h0['channels'], self.h0['kdim'], self.h0['stride'], self.h0['padding']),
            nn.LeakyReLU(self.leak)
        )
        # MNIST: Output should now be (-1, 128, 28, 28)


        self.hidden1 = nn.Sequential(
            nn.Conv2d(self.h0['channels'], self.h1['channels'], self.h1['kdim'], self.h1['stride'], self.h1['padding']),
            nn.LeakyReLU(self.leak)
        )
        # MNIST: Output should now be (-1, 128, 14, 14)

        self.hidden2 = nn.Sequential(
            nn.Conv2d(self.h1['channels'], self.h2['channels'], self.h2['kdim'], self.h2['stride'], self.h2['padding']),
            nn.LeakyReLU(self.leak)
        )
        # MNIST: Output should now be (-1, 64, 7, 7)

        self.hidden3 = nn.Sequential(
            nn.Conv2d(self.h2['channels'], self.h3['channels'], self.h3['kdim'], self.h3['stride'], self.h3['padding']),
            nn.LeakyReLU(self.leak)
        )
        # MNIST: Output should now be (-1, 64, 7, 7)

        self.out = nn.Sequential(
            nn.Linear(self.h1['channels']*(self.initial_dimensions[1]*self.initial_dimensions[2])//(total_down_fac**2),1),
            nn.Sigmoid()
        )
        # For an MNIST experiment output should now be (-1, 1, 28, 28)
        # Obviously depends on output_vector_dims otherwise

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x