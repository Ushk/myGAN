import torch
from torch.autograd.variable import Variable

import numpy as np

from tqdm import tqdm

from ganfuncs import  create_discriminator_input_and_labels, noise


class Experiment:

    def __init__(self, nepochs, data_loader, exp_step, observer=None):

        self.num_epochs = nepochs
        self.data_loader = data_loader

        self.experiment_step = exp_step

        self.gan_observer = observer



    def train_model(self):

        for epoch in range(self.num_epochs):

            self.experiment_step.reinitialize()

            for i, (inputs, labels) in enumerate(tqdm(self.data_loader, desc='Train', ascii=True)):

                self.experiment_step.step(i, inputs, labels)

                if (self.gan_observer is not None) & (epoch != 0):

                    if i%self.gan_observer.save_frequency==0:
                        self.gan_observer.save_model(self.experiment_step.gen_trainer.model,
                                                     self.experiment_step.dis_trainer.model)

                    self.gan_observer.update_training_metrics(self.experiment_step.log_dict)


            print(self.experiment_step.log_dict)


class GANExperimentStep:

    """
    Needs:
    1. Trainers
    2. Batch size
    3. Logger
    """

    def __init__(self, bs, generator_trainer, discriminator_trainer, CUDA):
        self.gen_trainer = generator_trainer
        self.dis_trainer = discriminator_trainer
        self.CUDA = CUDA

        # Logging Vars
        self.log_dict = {}


    def step(self, i, inputs, labels):

        batch_size = inputs.size(0)

        if self.CUDA:
            inputs = inputs.cuda()

        self.train_discriminator(batch_size, inputs)
        self.train_generator(batch_size)

        self.log_dict['loss_delta'] += np.abs(self.log_dict['gen_loss'].data - self.log_dict['dis_loss'].data)/batch_size


    def train_discriminator(self, batch_size, inputs):

        self.dis_trainer.optimizer.zero_grad()

        #Train Discriminator on Real Data
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(inputs, Variable(torch.ones(batch_size, 1)).cuda())

        # Train Discriminator on Fake Data
        generator_random_input = noise(batch_size)
        dis_fake_data = self.gen_trainer.model(generator_random_input.cuda())
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(dis_fake_data, Variable(torch.zeros(batch_size, 1)).cuda())
        self.dis_trainer.optimizer.step()

    def train_generator(self, batch_size):

        # Clear Gradients
        self.gen_trainer.optimizer.zero_grad()

        # Generate Fake Predictions
        generator_random_input = noise(2*batch_size)

        if self.CUDA:
            generator_random_input = generator_random_input.cuda()

        fake_data = self.gen_trainer.model(generator_random_input)
        fake_preds = self.dis_trainer.model(fake_data.cuda())

        # Calculate error and update
        error = self.gen_trainer.loss(fake_preds, Variable(torch.ones(2*batch_size, 1)).cuda())
        error.backward()
        self.log_dict['gen_loss'] += error
        self.gen_trainer.optimizer.step()


    def reinitialize(self):
        self.log_dict['gen_loss'] = 0
        self.log_dict['dis_loss'] = 0
        self.log_dict['loss_delta'] = 0

















