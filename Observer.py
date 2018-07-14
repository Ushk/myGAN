from tensorboardX import SummaryWriter
import torch
import os
import torchvision.utils as vutils

class Observable:
#
#     """
#     Base Class for Observable. Subclasses of this class will contain all of the information to be logged during the
#     course of an experiment
#     """
#
    def __init__(self):
        pass
#         self.writer = SummaryWriter()


class GANObserver(Observable):

    def __init__(self, save_frequency=5, save_loc='./runs/'):
        super(Observable, self).__init__()

        assert os.path.isdir(save_loc), "{} does not exist or is not a directory".format(save_loc)

        self.save_frequency = save_frequency

        self.run_path = save_loc+ 'run' + str(len(os.listdir(save_loc))+1)
        self.writer = SummaryWriter(self.run_path)
        self.cpt_path = self.run_path + '/cpts/'
        os.mkdir(self.cpt_path)


    def update_training_metrics(self, log_dict):
        """
        """
        for log_key, log_scalar in log_dict.items():
            self.writer.add_scalar(self.run_path +log_key, log_scalar)


    def save_model(self, generator_model, discriminator_model):
        torch.save(generator_model.state_dict(), self.cpt_path + 'gen_state_dict')
        torch.save(discriminator_model.state_dict(), self.cpt_path + 'dis_state_dict')

    def update_images(self, validation_images, epoch):
        x = vutils.make_grid(validation_images, normalize=True, scale_each=True)
        self.writer.add_image('Image', x, epoch)



