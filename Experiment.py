from tqdm import tqdm


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
                        val_images = self.experiment_step.validation_step()
                        self.gan_observer.update_images(val_images, epoch)

                    self.gan_observer.update_training_metrics(self.experiment_step.log_dict)


            print(self.experiment_step.log_dict)


















