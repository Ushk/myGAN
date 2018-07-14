import torch
import os

from torchvision import transforms, datasets
from torch import nn, optim
from Generators import  SimpleGenerator
from Discriminators import SimpleDiscriminator
from Trainer import Trainer

from Experiment import Experiment
from ExperimentStep import GANExperimentStep
from Observer import GANObserver

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_FEATS = 100
MNIST_DIM = 784
IS_CONV = False
LOG_RUNS = True
CUDA = True
lr = 0.0002

def mnist_data():
    transform_list = [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
    if not IS_CONV:
        transform_list.append(transforms.Lambda(lambda x: x.view(MNIST_DIM)))
    compose = transforms.Compose(transform_list)
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# Create Networks
gen = SimpleGenerator(NUM_FEATS, MNIST_DIM, leak=0.1)
dis = SimpleDiscriminator(MNIST_DIM,1, leak=0.3, drop_out=0.2)


# Create Trainers
gen_trainer = Trainer(gen, lr)
dis_trainer = Trainer(dis, lr)

GAN_observer = None
if LOG_RUNS is True:
    GAN_observer = GANObserver()


exp_step = GANExperimentStep(BATCH_SIZE, gen_trainer, dis_trainer)
exp = Experiment(nepochs=NUM_EPOCHS, data_loader=data_loader, exp_step=exp_step, observer=GAN_observer)
exp.train_model()





