from tensorboardX import SummaryWriter
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

    def __init__(self, save_loc):
        super(Observable, self).__init__()
        self.writer = SummaryWriter()
        self.save_loc = save_loc

    def update(self, generator_loss, discriminator_loss, loss_delta, images=None):
        """
        """
        self.writer.add_scalar(self.save_loc +'gen_loss',generator_loss)
        self.writer.add_scalar(self.save_loc +'dis_loss',discriminator_loss)
        self.writer.add_scalar(self.save_loc +'loss_del',loss_delta)
        # if images is not None:
        #     for img_idx, img in enumerate(images):
        #         self.writer.add_image(self.save_loc +'Example_Image_' + str(img_idx), img)

