import torch
from torch.autograd.variable import Variable

import numpy as np
from ganfuncs import  create_discriminator_input_and_labels, noise

class GANExperimentStep:

    """
    Needs:
    1. Trainers
    2. Batch size
    3. Logger
    """

    def __init__(self, bs, generator_trainer, discriminator_trainer):
        self.gen_trainer = generator_trainer
        self.dis_trainer = discriminator_trainer

        # Logging Vars
        self.log_dict = {}


    def step(self, i, inputs, labels):

        batch_size = inputs.size(0)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        self.train_discriminator(batch_size, inputs)
        self.train_generator(batch_size)

        self.log_dict['loss_delta'] += np.abs(self.log_dict['gen_loss'].data - self.log_dict['dis_loss'].data)/batch_size


    def train_discriminator(self, batch_size, inputs):

        self.dis_trainer.optimizer.zero_grad()

        if torch.cuda.is_available():
            real_targets = Variable(torch.ones(batch_size, 1)).cuda()
            fake_targets = Variable(torch.zeros(batch_size, 1)).cuda()
            generator_random_input = noise(batch_size).cuda()

        else:
            real_targets = Variable(torch.ones(batch_size, 1))
            fake_targets = Variable(torch.zeros(batch_size, 1))
            generator_random_input = noise(batch_size)

        #Train Discriminator on Real Data
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(inputs, real_targets)

        # Train Discriminator on Fake Data
        dis_fake_data = self.gen_trainer.model(generator_random_input.cuda())
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(dis_fake_data, fake_targets)

        #Update Discriminator Weights
        self.dis_trainer.optimizer.step()

    def train_generator(self, batch_size):

        # Clear Gradients
        self.gen_trainer.optimizer.zero_grad()

        # Generate Fake Predictions
        if torch.cuda.is_available():
            generator_random_input = noise(2 * batch_size).cuda()
            fake_data = self.gen_trainer.model(generator_random_input).cuda()
            generator_targets = Variable(torch.ones(2*batch_size, 1)).cuda()

        else:
            generator_random_input = noise(2 * batch_size)
            fake_data = self.gen_trainer.model(generator_random_input)
            generator_targets = Variable(torch.ones(2 * batch_size, 1))


        fake_preds = self.dis_trainer.model(fake_data)

        # Calculate error and update
        error = self.gen_trainer.loss(fake_preds, generator_targets)
        error.backward()
        self.log_dict['gen_loss'] += error
        self.gen_trainer.optimizer.step()


    def reinitialize(self):
        self.log_dict['gen_loss'] = 0
        self.log_dict['dis_loss'] = 0
        self.log_dict['loss_delta'] = 0

    def validation_step(self, test_outputs=10):

        if torch.cuda.is_available():
            generator_random_input = noise(test_outputs).cuda()
            fake_data = self.gen_trainer.model(generator_random_input).cuda()
        else:
            generator_random_input = noise(test_outputs).cuda()
            fake_data = self.gen_trainer.model(generator_random_input).cuda()

        fake_data = fake_data.view(-1,1,28, 28).data
        return fake_data



