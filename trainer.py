import torch
import torch.nn as nn
import torch.functional as F




class Trainer:

    def __init__(self, optim, loss, model):
        """
        Trainer class. The trainer class *has a* model. It handles the training logic. It should not need to know
        the internals of the model.
        :param optim: E.g. SGD/ADAM.
        :param loss:
        :param model:
        """
        self.optimizer = optim
        self.loss = loss
        self.model = model

    def train_step(self, predictions, labels, retain=False):
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

        self.optimizer.zero_grad()

        error = self.loss(predictions, labels)

        error.backward(retain_graph=retain)

        self.optimizer.step()

        return error
