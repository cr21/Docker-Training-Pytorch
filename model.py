import torch
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, ReLU, LogSoftmax, Flatten


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # create model here

    def forward(self, x):
        return self.main(x)
