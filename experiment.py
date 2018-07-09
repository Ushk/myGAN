import torch
from torch.autograd.variable import Variable

import numpy as np
import os

from tqdm import tqdm

from observer import GANObservable
from ganfuncs import  create_discriminator_input_and_labels, noise


class Experiment:

    def __init__(self, nepochs, data_loader,exp_step, save_loc='./runs', save_freq=10):

        assert os.path.isdir(save_loc), "{} does not exist or is not a directory".format(save_loc)

        self.num_epochs = nepochs
        self.save_loc = save_loc
        self.data_loader = data_loader


        self.experiment_step = exp_step
        self.save_loc = save_loc
        self.save_freq = save_freq


    def train_model(self):

        for epoch in range(self.num_epochs):

            self.experiment_step.reinitialize()

            for i, (inputs, labels) in enumerate(tqdm(self.data_loader, desc='Train', ascii=True)):

                self.experiment_step.step(i, inputs, labels)

                if i%self.save_freq==0:
                    self.experiment_step.save(self.save_loc)

            print(self.experiment_step.log_dict)


class GANExperimentStep:

    """
    Needs:
    1. Trainers
    2. Batch size
    3. Logger
    """

    def __init__(self,bs, generator_trainer, discriminator_trainer, step_freq, num_sample_imgs=5, save_loc='./runs/logs/'):
        self.gen_trainer = generator_trainer
        self.dis_trainer = discriminator_trainer

        # Logging Vars
        self.log_step_freq = step_freq
        self.logger = GANObservable(save_loc)
        self.log_dict = {}


    def step(self, i, inputs, labels):

        batch_size = inputs.size(0)

        self.train_discriminator(batch_size, inputs)
        self.train_generator(batch_size)

        self.log_dict['loss_delta'] += np.abs(self.log_dict['gen_loss'].data - self.log_dict['dis_loss'].data)/batch_size


    def train_discriminator(self, batch_size, inputs):

        self.dis_trainer.optimizer.zero_grad()

        #Train Discriminator on Real Data
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(inputs, Variable(torch.ones(batch_size, 1)))

        # Train Discriminator on Fake Data
        generator_random_input = noise(batch_size)
        dis_fake_data = self.gen_trainer.model(generator_random_input)
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(dis_fake_data, Variable(torch.zeros(batch_size, 1)))
        self.dis_trainer.optimizer.step()

    def train_generator(self, batch_size):

        # Clear Gradients
        self.gen_trainer.optimizer.zero_grad()

        # Generate Fake Predictions
        generator_random_input = noise(2*batch_size)
        fake_data = self.gen_trainer.model(generator_random_input)
        fake_preds = self.dis_trainer.model(fake_data)

        # Calculate error and update
        error = self.gen_trainer.loss(fake_preds, Variable(torch.ones(2*batch_size, 1)))
        error.backward()
        self.log_dict['gen_loss'] += error
        self.gen_trainer.optimizer.step()


    def save(self, save_loc='./runs'):
        torch.save(self.gen_trainer.model.state_dict(), save_loc+'gen_state_dict')
        torch.save(self.dis_trainer.model.state_dict(), save_loc+'dis_state_dict')

    def reinitialize(self):
        self.log_dict['gen_loss'] = 0
        self.log_dict['dis_loss'] = 0
        self.log_dict['loss_delta'] = 0

















