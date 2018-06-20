import torch
from torch.autograd.variable import Variable

import numpy.random as nr
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

            for i, (inputs, labels) in enumerate(tqdm(self.data_loader, desc='Train', ascii=True)):

                self.experiment_step.step(i, inputs, labels)

                if i%self.save_freq==0:
                    self.experiment_step.save(self.save_loc)


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
        self.batch_size = bs
        self.gen_targets = Variable(torch.ones(self.batch_size, 1))

        # Logging Vars
        self.log_step_freq = step_freq
        self.logger = GANObservable(save_loc)
        self.log_dict = {}
        self.num_imgs = num_sample_imgs



    def step(self, i, inputs, labels):

        generator_random_input = noise(self.batch_size)

        self.gen_trainer.model.train()
        self.dis_trainer.model.train()

        fake_data = self.gen_trainer.model(generator_random_input)

        discriminator_input, discriminator_labels = create_discriminator_input_and_labels(fake_data, inputs, self.batch_size)

        discriminator_predictions = self.dis_trainer.model(discriminator_input)

        # Train the two networks, and get their loss
        self.log_dict['generator_loss'] = self.gen_trainer.train_step(discriminator_predictions,self.gen_targets, retain=True)

        self.log_dict['discriminator_loss'] = self.dis_trainer.train_step(discriminator_predictions, discriminator_labels, retain=False)

        # Calculate the difference in loss between the two networks; should be close to 0 or we have a collapse.
        self.log_dict['loss_delta'] = self.log_dict['discriminator_loss'] - self.log_dict['generator_loss']

        # At certain frequencies, get example images
        if i % self.log_step_freq ==0:
            rand_inds = nr.randint(0, self.batch_size, size=self.num_imgs)
            self.log_dict['images'] = fake_data[rand_inds,:]

        # Update the logging class
        self.logger.update(**self.log_dict)

    def save(self, save_loc='./runs'):
        torch.save(self.gen_trainer.model.state_dict(), save_loc+'gen_state_dict')
        torch.save(self.dis_trainer.model.state_dict(), save_loc+'dis_state_dict')













