import torch
from torch import nn, optim

from torch.autograd.variable import Variable


class Trainer:

    def __init__(self, model, lr, gpu=None):
        """
        Trainer class. The trainer class *has a* model. It handles the training logic. It should not need to know
        the internals of the model.
        :param optim: E.g. SGD/ADAM.
        :param loss:
        :param model:
        """
        self.loss = nn.BCELoss()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.model = self.model.cuda()



    def train_step(self, data, labels):
        """
        1. Receives predictions from ExperimentStep class
        2. Computes loss
        3. Takes a backward step
        4. Returns error/accuracy
        Again, this function should not need to know the internals of the model. It should simply receive data and
        update the weights
        :param predictions: appropriate data for the model
        :param labels: appropriate data for the model
        :return: error: value of the loss on the data
        """

        predictions = self.model(data)

        error = self.loss(predictions, labels)

        error.backward()

        return error











