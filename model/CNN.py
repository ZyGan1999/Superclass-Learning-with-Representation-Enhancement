import torch
import torchvision
import numpy as np
import torch.nn.functional as F

class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 6,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.fc1 = torch.nn.Linear(16*5*5,120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,17)
        self.fc4 = torch.nn.Linear(10,2)

    def forward(self,x):
        out=self.pool(F.relu(self.conv1(x)))
        out=self.pool(F.relu(self.conv2(out)))
        out = out.reshape(x.shape[0], -1)
        feature = out
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        cls_rtn = F.relu(self.fc3(out))
        return feature, cls_rtn

