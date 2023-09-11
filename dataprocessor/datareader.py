import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import hyperparameters as HP

'''
def get_CIFAR10_dataloader():
    
    read CIFAR10 data from the directory '../data'
    return the dataloader of the training set and test set

    50000 instances for the training set
    10000 instances for the test set
    
    for each image instance, the size is (3, 32, 32)
    for each batch, the size is (batch_size, 3, 32, 32)
    
    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=HP.batch_size,shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=HP.batch_size,shuffle=False, num_workers=2)

    print('ok')

    return trainloader, testloader
'''


def get_CIFAR10_dataloader():
    '''
    read CIFAR10 data from the directory '../data'
    return the dataloader of the training set and test set

    50000 instances for the training set
    10000 instances for the test set
    
    for each image instance, the size is (3, 32, 32)
    for each batch, the size is (batch_size, 3, 32, 32)
    '''

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    machine = (0,1,8,9)
    animal = (2,3,4,5,6,7)

    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    #trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

    
    train_data_X = []
    train_data_Y = []
    test_data_X = []
    test_data_Y = []

    for x, y in tqdm(trainset):
        train_data_X.append(x)
        if y in machine:
            train_data_Y.append(0)
        else:
            train_data_Y.append(1)

    for x, y in tqdm(testset):
        test_data_X.append(x)
        if y in machine:
            test_data_Y.append(0)
        else:
            test_data_Y.append(1)
    train_data_X = torch.stack(train_data_X, dim=0)
    train_data_Y = torch.tensor(train_data_Y)

    test_data_X = torch.stack(test_data_X, dim=0)
    test_data_Y = torch.tensor(test_data_Y)

    print(train_data_X.size(), train_data_Y.size())
    print(test_data_X.size(), test_data_Y.size())

    train_data_set = TensorDataset(train_data_X, train_data_Y)
    test_data_set = TensorDataset(test_data_X, test_data_Y)

    train_loader = DataLoader(dataset=train_data_set, batch_size=HP.batch_size, shuffle=True, num_workers=2,drop_last=True)
    test_loader = DataLoader(dataset=test_data_set, batch_size=HP.batch_size, shuffle=True, num_workers=2,drop_last=True)

    return train_loader, test_loader




'''
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# dataiter=iter(trainloader)
# images,labels=dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(''.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    dataiter=iter(trainloader)
    images,labels=dataiter.next()

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
'''
