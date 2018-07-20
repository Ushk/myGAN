import torch
import os

from torchvision import transforms, datasets

from Generators import  SimpleGenerator, ConvolutionalGenerator
from Discriminators import SimpleDiscriminator, ConvolutionalDiscriminator
from Trainer import Trainer

from Experiment import Experiment
from ExperimentStep import GANExperimentStep
from Observer import GANObserver
from NetworkProperties import NetworkProperties

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
NUM_EPOCHS = 100
BATCH_SIZE = 64
LOG_RUNS =True 
lr = 0.0002

def mnist_data():
    transform_list = [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
    compose = transforms.Compose(transform_list)
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# Create Networks
NUM_FEATS = (1,10,10)
MNIST_DIM=(1,28,28)

gen = ConvolutionalGenerator(NUM_FEATS, MNIST_DIM, leak=0.1)
dis = ConvolutionalDiscriminator(MNIST_DIM,1, leak=0.3)


# Create Trainers
gen_trainer = Trainer(gen, lr)
dis_trainer = Trainer(dis, lr)

GAN_observer = None
if LOG_RUNS is True:
    GAN_observer = GANObserver(1)


exp_step = GANExperimentStep(gen_trainer, dis_trainer, NUM_FEATS)
exp = Experiment(nepochs=NUM_EPOCHS, data_loader=data_loader, exp_step=exp_step, observer=GAN_observer)
exp.train_model()





