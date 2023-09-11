# -*- coding: utf-8 -*-
import torch
import torchvision.datasets as dataset
from torchvision import transforms
import pandas as pd

def CIFAR100_4(data, label_file):
    data = list(data)
    to_tensor = transforms.ToTensor()
    for i in range(len(data)):
        data[i] = list(data[i])
        #data[i][0] = transforms.ToTensor(data[i][0])
        data[i][0] = to_tensor(data[i][0])
        data[i][1] = label_file.iloc[data[i][1],2]
    return data

if __name__ == '__main__':
    CIFAR100train = dataset.CIFAR100(root='./', train=True, download=True, transform=None)
    CIFAR100test = dataset.CIFAR100(root='./', train=False, download=True, transform=None)
    label_file = pd.read_csv('./CIFAR100-4.csv')
    #torch.save(CIFAR100_4(CIFAR100train, label_file),'./train.pt')
    #torch.save(CIFAR100_4(CIFAR100test, label_file),'./test.pt')
    torch.save(CIFAR100_4(CIFAR100train, label_file),'../data/re_cifar100/4_categories/train.pt')
    torch.save(CIFAR100_4(CIFAR100test, label_file),'../data/re_cifar100/4_categories/test.pt')
    print("Superclass Dataset Generated.")