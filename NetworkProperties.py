import torch.nn as nn

class NetworkProperties:

    def __init__(self, input_dim, output_dim,learning_rate, dropout=1.0, leak=0.0):
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        self.lr = learning_rate,
        self.drop_out = dropout
        self.leak = leak
