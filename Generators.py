import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SimpleGenerator(nn.Module):

    def __init__(self, nfeatures, n_out, leak):
        super(SimpleGenerator, self).__init__()

        self.g1_neurons = 256
        self.g2_neurons = 512
        self.g3_neurons = 1024

        self.leak = leak

        assert type(nfeatures)==tuple, 'The nfeatures parameter should be a tuple with the input vector dimensions,' \
                                       'currently has type'.format(type(tuple))
        self.nfeatures = int(np.prod(nfeatures[1:]))
        self.output_shape = n_out
        self.noutput_neurons = int(np.prod(n_out[1:]))  # TODO - Hacky, can this be fixed?

        self.hidden0 = nn.Sequential(
            nn.Linear(self.nfeatures, self.g1_neurons),
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
            nn.Linear(self.g3_neurons, self.noutput_neurons),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.nfeatures)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        x = x.view(-1, *self.output_shape)
        return x


class ConvolutionalGenerator(nn.Module):

    def __init__(self, input_vector_dims, output_vector_dims, leak):
        """

        :param input_vector_dims: tuple with dimensions ( C, H, W)
        :param output_vector_dims: tuple with desired output dimensions ( C, H, W) - e.g. (1,28,28) for MNIST
        :param leak: leak for leaky RELU
        """

        super(ConvolutionalGenerator, self).__init__()

        self.initial_dimensions = input_vector_dims
        self.output_vector_dims = output_vector_dims

        self.kdim = 3

        self.h0 = {'channels': 128, 'kdim': self.kdim, 'stride': 1, 'padding': 0}
        self.h1 = {'channels': 64, 'kdim': self.kdim, 'stride': 1, 'padding': 0}
        self.h2 = {'channels': 32, 'kdim': self.kdim, 'stride': 2, 'padding': 1}
        self.h3 = {'channels': self.output_vector_dims[0], 'kdim': self.kdim, 'stride': 2, 'padding': 1}

        self.leak = leak

        self.hidden0 = nn.Sequential(
            nn.ConvTranspose2d(self.initial_dimensions[0], self.h0['channels'], self.h0['kdim'], self.h0['stride'], self.h0['padding']),
            nn.LeakyReLU(self.leak),
            nn.BatchNorm2d(self.h0['channels'])
        )
        # Assuming an input with (H,W) = (1,1), activations should now be (-1, 512, 3, 3)

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(self.h0['channels'], self.h1['channels'], self.h1['kdim'], self.h1['stride'], self.h1['padding']),
            nn.LeakyReLU(self.leak),
            nn.BatchNorm2d(self.h1['channels'])
        )
        # Output should now be (-1, 256, 6, 6)

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(self.h1['channels'], self.h2['channels'], self.h2['kdim'], self.h2['stride'], self.h2['padding']),
            nn.LeakyReLU(self.leak),
            nn.BatchNorm2d(self.h2['channels'])

        )
        # Output should now be (-1, 128, 12, 12)

        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(self.h2['channels'], self.h3['channels'], self.h2['kdim'], self.h2['stride'], self.h2['padding']),
            nn.LeakyReLU(self.leak),
            nn.BatchNorm2d(self.h3['channels'])
        )

        self.out = nn.Sequential(
            nn.Upsample(size=output_vector_dims[1:], mode='bilinear', align_corners = False),
            nn.Tanh()
        )
        # For an MNIST experiment output should now be (-1, 1, 28, 28)
        # Obviously depends on output_vector_dims otherwise

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x