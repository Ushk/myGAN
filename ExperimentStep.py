import torch
from torch.autograd.variable import Variable
from ganfuncs import  create_discriminator_input_and_labels, noise

class GANExperimentStep:

    """
    Needs:
    1. Trainers
    2. Batch size
    3. Logger
    """

    def __init__(self, generator_trainer, discriminator_trainer, features):
        self.gen_trainer = generator_trainer
        self.dis_trainer = discriminator_trainer
        self.epoch = 0

        # Noise shape for generator
        # If 1, should be passed to NN with linear first layer, i.e the data is flattened
        # If 3, should have format (C,H,W)
        assert len(tuple(features)) in (1,3), 'Invalid Feature dimension'
        self.input_features = features

        # Logging Vars
        self.log_dict = {}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() is not False else "cpu")


    def step(self, epoch, inputs, labels):
        self.epoch = epoch + 1

        batch_size = inputs.size(0)

        inputs = inputs.to(self.device)

        self.train_discriminator(batch_size, inputs)
        self.train_generator(batch_size)

        self.log_dict['loss_delta'] += (self.log_dict['gen_loss'].data - self.log_dict['dis_loss'].data)/batch_size


    def train_discriminator(self, batch_size, inputs):

        self.dis_trainer.optimizer.zero_grad()


        real_targets = Variable(torch.ones(batch_size, 1, device=self.device))
        fake_targets = Variable(torch.zeros(batch_size, 1, device=self.device))
        generator_random_input = Variable(torch.randn(batch_size, *self.input_features, device=self.device))

        inputs = max(0, (10-self.epoch)/self.epoch)*torch.randn_like(inputs) + inputs

        #Train Discriminator on Real Data
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(inputs, real_targets)/batch_size

        # Train Discriminator on Fake Data
        dis_fake_data = self.gen_trainer.model(generator_random_input)
        dis_fake_data = max(0, (10 - self.epoch) / self.epoch) * torch.randn_like(dis_fake_data) + dis_fake_data
        self.log_dict['dis_loss'] += self.dis_trainer.train_step(dis_fake_data, fake_targets)/batch_size

        #Update Discriminator Weights
        self.dis_trainer.optimizer.step()

    def train_generator(self, batch_size):

        # Clear Gradients
        self.gen_trainer.optimizer.zero_grad()

        # Generate Fake Predictions
        generator_random_input = Variable(torch.randn(2*batch_size, *self.input_features, device=self.device))
        fake_data = self.gen_trainer.model(generator_random_input)
        generator_targets = Variable(torch.ones(2 * batch_size, 1, device=self.device) )


        fake_preds = self.dis_trainer.model(fake_data)

        # Calculate error and update
        error = self.gen_trainer.loss(fake_preds, generator_targets)
        error.backward()
        self.log_dict['gen_loss'] += error/batch_size
        self.gen_trainer.optimizer.step()


    def reinitialize(self):
        self.log_dict['gen_loss'] = 0
        self.log_dict['dis_loss'] = 0
        self.log_dict['loss_delta'] = 0

    def validation_step(self, test_outputs=10):


        generator_random_input = Variable(torch.randn(test_outputs, *self.input_features, device=self.device))
        fake_data = self.gen_trainer.model(generator_random_input)

        # fake_data = fake_data.view(-1,1,28, 28).data
        return fake_data



