import numpy.random as nr
import torch

from torch.autograd.variable import Variable


def create_discriminator_input_and_labels(fake_data, real_data, batch_size):
    # Generate indices to sample from fake_data (generator predictions)
    sampled_fake_data_indices = nr.choice(batch_size, size=batch_size // 2, replace=False)

    # Generate indices to sample from real_data (training data)
    sampled_real_data_indices = nr.choice(batch_size, size=batch_size // 2, replace=False)

    # Generate indices for final shuffle (we cannot use .shuffle, because we need the shuffling to be the same for both
    # inputs and labels)
    final_indices = nr.choice(batch_size, size=batch_size, replace=False)

    # Generate fake/real labels for data
    sampled_fake_labels = torch.zeros(batch_size // 2)
    sampled_real_labels = torch.ones(batch_size // 2)

    # Concatenate the samples from the two datasets, and their corresponding labels
    disc_input = torch.cat((fake_data[sampled_fake_data_indices,:],real_data[sampled_real_data_indices,:]),0)
    disc_labels = torch.cat((sampled_fake_labels, sampled_real_labels),0)

    # Shuffle the data
    disc_input=disc_input[final_indices]
    disc_labels=disc_labels[final_indices]

    return disc_input, disc_labels

def noise(size):
    '''
    Generates a 1D vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n