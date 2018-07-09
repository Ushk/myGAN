import torch

from torch.autograd.variable import Variable


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


class DiscriminatorTrainer:

    def __init__(self, optim, loss, model):
        self.optimizer = optim
        self.loss = loss
        self.model = model

    def calc_error(self, data, labs):

        preds = self.model(data)
        error = self.loss(preds, labs)
        error.backward()


    def train_step(self, real_data, fake_data, batch_size):
        real_labs = Variable(torch.ones(batch_size, 1))
        fake_labs = Variable(torch.zeros(batch_size, 1))

        self.optimizer.zero_grad()

        self.calc_error(real_data, real_labs)
        self.calc_error(fake_data, fake_labs)

        self.optimizer.step()



