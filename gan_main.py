import torch
import os

from torchvision import transforms, datasets
from torch import nn, optim

from trainer import Trainer
from networks import SimpleDiscriminator, SimpleGenerator
from experiment import Experiment, GANExperimentStep

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_FEATS = 100
MNIST_DIM = 784
IS_CONV = False
LOG_RUNS = False


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

# Optimizers
d_optimizer = optim.Adam(dis.parameters(), lr=0.0002)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002)

loss = nn.BCELoss()

# Create Trainers
gen_trainer = Trainer(g_optimizer, loss, gen)
dis_trainer = Trainer(d_optimizer, loss, dis)


if LOG_RUNS is True:
    # Create Save locations
    run_path = './runs/'+ 'run' + str(len(os.listdir('./runs'))+1)
    cpt_path = run_path+'/cpts/'
    log_path = run_path+'/logs/'

    os.mkdir(run_path)
    os.mkdir(cpt_path)
    os.mkdir(log_path)

    exp_step = GANExperimentStep(BATCH_SIZE, gen_trainer, dis_trainer, step_freq=5, save_loc=log_path)
    exp = Experiment(nepochs=NUM_EPOCHS, data_loader=data_loader, exp_step=exp_step,save_loc=cpt_path)

    exp.train_model()





