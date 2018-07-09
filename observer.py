from tensorboardX import SummaryWriter
import torch
import os

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


class GANObservable(Observable):

    def __init__(self, logging_on, save_frequency, save_loc):
        super(Observable, self).__init__()

        assert os.path.isdir(save_loc), "{} does not exist or is not a directory".format(save_loc)

        self.logging = logging_on

        self.save_frequency = save_frequency
        self.writer = SummaryWriter()
        self.save_loc = save_loc


    def make_dirs(self):

        if self.logging is True:
            run_path = './runs/'+ 'run' + str(len(os.listdir('./runs'))+1)
            cpt_path = run_path+'/cpts/'
            log_path = run_path+'/logs/'

            os.mkdir(run_path)
            os.mkdir(cpt_path)
            os.mkdir(log_path)



    def update_training_metrics(self, generator_loss, discriminator_loss, loss_delta, images=None):
        """
        """
        if self.logging is True:
            self.writer.add_scalar(self.save_loc +'gen_loss',generator_loss)
            self.writer.add_scalar(self.save_loc +'dis_loss',discriminator_loss)
            self.writer.add_scalar(self.save_loc +'loss_del',loss_delta)
            # if images is not None:
            #     for img_idx, img in enumerate(images):
            #         self.writer.add_image(self.save_loc +'Example_Image_' + str(img_idx), img)

    def save_model(self, generator_model, discriminator_model):
        if self.logging is True:
            torch.save(generator_model.state_dict(), self.save_loc + 'gen_state_dict')
            torch.save(discriminator_model.state_dict(), self.save_loc + 'dis_state_dict')



