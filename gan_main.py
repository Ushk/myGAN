import torch
import os

from torchvision import transforms, datasets
from torch import nn, optim

from trainer import Trainer
from networks import SimpleDiscriminator, SimpleGenerator
from experiment import Experiment, GANExperimentStep
from observer import GANObserver

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_FEATS = 100
MNIST_DIM = 784
IS_CONV = False
LOG_RUNS = False
CUDA = False


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

if CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gen = gen.cuda()
    dis = dis.cuda()


# Optimizers
d_optimizer = optim.Adam(dis.parameters(), lr=0.0002)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002)

loss = nn.BCELoss()

# Create Trainers
gen_trainer = Trainer(g_optimizer, loss, gen)
dis_trainer = Trainer(d_optimizer, loss, dis)

GAN_observer = None
if LOG_RUNS is True:
    GAN_observer = GANObserver()


exp_step = GANExperimentStep(BATCH_SIZE, gen_trainer, dis_trainer, CUDA)
exp = Experiment(nepochs=NUM_EPOCHS, data_loader=data_loader, exp_step=exp_step, observer=GAN_observer)
exp.train_model()





