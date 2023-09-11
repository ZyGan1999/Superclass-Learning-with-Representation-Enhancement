from cProfile import label
from cgi import test
import imp
from locale import normalize
from random import shuffle
import random
from re import X
from dataprocessor.datareader import get_CIFAR10_dataloader
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Resize
import hyperparameters as HP
import os
from torchvision import transforms
from PIL import Image
from utils import single_channel_to_3_channel
from utils import mini_imagenet_Dataset
from utils import ut_zap50k_Dataset
from utils import CIFAR100Pair
from utils import train_transform
from utils import test_transform
import FMoW
import iWildCam

def get_CIFAR10_re_dataset():
    '''
    based on the specific task, re-organize the labels of the CIFAR10 data
    divide the instances into 2 categories, where 0 for 'machine', 1 for 'animal'
    return the dataloader and the new labels (i.e. macro labels) on both the training set and test set

    the size of the macro_labels is (batch_num, 1), where batch_num = instance_num / batch_size
    '''


    trainloader, testloader = get_CIFAR10_dataloader() # get the dataloader (batchsize=HP.batch_size)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    machine = (0,1,8,9)
    animal = (2,3,4,5,6,7)
    train_macro_labels = []
    test_macro_labels = []

    print('operating training data...')
    for images, labels in trainloader:
        batch_labels = []
        for label in labels:
            if label in machine:
                batch_labels.append(0)
            else:
                batch_labels.append(1)
        train_macro_labels.append(batch_labels)
    print('traning data finished.')
    train_macro_labels = np.array(train_macro_labels)

    print('operating test data...')
    for images, labels in testloader:
        batch_labels = []
        for label in labels:
            if label in machine:
                batch_labels.append(0)
            else:
                batch_labels.append(1)
        test_macro_labels.append(batch_labels)
    print('test data finished.')
    test_macro_labels = np.array(test_macro_labels)


    return trainloader, testloader, train_macro_labels, test_macro_labels


def get_101_OC_data():
    x = torch.load('./data/101_Object_Categories/data/data.pt')
    y = torch.load('./data/101_Object_Categories/data/macro_label.pt')

    print(x.size(), y.size())

    deal_dataset = TensorDataset(x,y)

    length=len(deal_dataset)
    print(length)
    train_size,test_size=int(0.8*length),int(0.2*length)
    if train_size + test_size != length:
        train_size += length - (train_size + test_size)
    #first param is data set to be saperated, the second is list stating how many sets we want it to be.
    train_set,test_set=torch.utils.data.random_split(deal_dataset,[train_size,test_size])
    #print(train_set,validate_set)
    print(len(train_set),len(test_set))

    train_loader = DataLoader(dataset=train_set, batch_size=HP.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=HP.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader

def get_101_data_split_by_macro_label(is_enumerate = False):
    root_dir = './data/101_data_split_by_macro_label'

    label_list = []

    label_to_idx = {}

    train_data_loader_dict = {}

    #train_data_dataloader_iter_dict = {}

    test_data_tensors_list = []
    test_data_labels_list = []

    idx = 0
    for item in os.listdir(root_dir):
        # to load the data.
        x = torch.load(root_dir + '/' + item)

        # to split the data to train and test set.
        x_train, x_test = torch.split(x, int(0.8*x.size()[0]), dim=0)

        # to create the label, label_to_idx, and label_list.
        label = item.replace('.pt','')
        label_to_idx[label] = idx
        label_list.append(label)
        
        # to record the test data tensors in list.
        test_data_tensors_list.append(x_test)

        # to record the test data labels in list.
        for i in range(x_test.size()[0]):
            test_data_labels_list.append(label_to_idx[label])
        
        # to record the train data tensors in dict by using label as keys.
        cur_train_label_list = []
        for i in range(x_train.size()[0]):
            cur_train_label_list.append(label_to_idx[label])
        train_data_dataset = TensorDataset(x_train, torch.tensor(cur_train_label_list))
        if is_enumerate:
            train_data_dataloader = DataLoader(dataset = train_data_dataset, batch_size = 1, shuffle = True, num_workers = 2)
        else:
            train_data_dataloader = DataLoader(dataset = train_data_dataset, batch_size = HP.batch_size, shuffle = True, num_workers = 2)

        train_data_loader_dict[label] = train_data_dataloader
        #train_data_dataloader_iter_dict[label] = iter(train_data_dataloader)

        idx += 1

    # to make the test data and label to tensor
    test_data_tensors = torch.cat(test_data_tensors_list, dim=0)
    test_data_labels = torch.tensor(test_data_labels_list)

    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_data_labels), batch_size = HP.batch_size, shuffle = True, num_workers = 2)

    #for item in label_to_idx.keys():
    #    print(item)


    return label_to_idx, train_data_loader_dict, test_data_loader


def get_CIFAR100_test_sample_tensor():
    training_set = torch.load('./data/CIFAR100/test.pt')
    Xs = []
    Ys = []
    for img, label in training_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(10000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors


def get_CIFAR100_data_loader():
    training_set = torch.load('./data/CIFAR100/training.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/CIFAR100/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader


def get_test_data_loader():
    training_set = torch.load('./data/test/2_categories_train280_test120/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        #print(img)
        #print(label)
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        torch_resize = Resize([84,84]) # 定义Resize类对象
        x = torch_resize(img)
        #print(x)
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/test/2_categories_train280_test120/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        torch_resize = Resize([84,84]) # 定义Resize类对象
        x = torch_resize(img)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_test_test_sample_tensor():
    training_set = torch.load('./data/test/2_categories_train280_test120/test.pt')
    Xs = []
    Ys = []
    for img, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        torch_resize = Resize([84,84]) # 定义Resize类对象
        x = torch_resize(img)
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(120), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors


def get_fake_data_loader():
    training_set = torch.load('./data/fake/fake_train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        #print(img)
        #print(label)
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x=img
        #print(x)
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/fake/real_test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x=img
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_fake_test_sample_tensor():
    training_set = torch.load('./data/fake/real_test.pt')
    Xs = []
    Ys = []
    for img, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x=img
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(400), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_FashionMNIST_data_loader():
    training_set = torch.load('./data/FashionMNIST/training.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        img = img.convert("RGB")
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/FashionMNIST/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        img = img.convert("RGB")
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_VOC_data_loader():
    training_set = torch.load('./data/VOC/training.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for img, label in training_set:
        img = img.resize((300, 300),Image.ANTIALIAS)
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/VOC/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for img, label in test_set:
        img = img.resize((300, 300),Image.ANTIALIAS)
        x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_MNIST_arc_data_loader():
    training_set = torch.load('./data/MNIST_arc/training.pt')
    img, label = training_set

    training_data_tensors = single_channel_to_3_channel(img) # 60000
    training_label_tensors = label

    training_data_tensors = training_data_tensors.float()
    
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    test_set = torch.load('./data/MNIST_arc/test.pt')
    img, label = test_set

    test_data_tensors = single_channel_to_3_channel(img) # 10000
    test_label_tensors = label

    test_data_tensors = test_data_tensors.float()
    
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader


def get_MNIST_orientation_data_loader():
    training_set = torch.load('./data/MNIST_orientation/training.pt')
    img, label = training_set

    training_data_tensors = single_channel_to_3_channel(img) # 60000
    training_label_tensors = label

    training_data_tensors = training_data_tensors.float()
    
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    test_set = torch.load('./data/MNIST_orientation/test.pt')
    img, label = test_set

    test_data_tensors = single_channel_to_3_channel(img) # 10000
    test_label_tensors = label

    test_data_tensors = test_data_tensors.float()
    
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print('training tensors: ', training_data_tensors.size(), training_label_tensors.size())
    print('test tensors: ', test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_mini_imagenet_animal_data_loader():
    train_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_animal_or_not/train.csv')
    test_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_animal_or_not/test.csv')

    train_loader = DataLoader(dataset=train_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)

    print('train and test size: ', len(train_data), len(test_data))

    return train_loader, test_loader

def get_mini_imagenet_animal_test_sample_tensor():
    test_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_animal_or_not/test.csv')
    test_loader = DataLoader(dataset=test_data, batch_size = HP.sample_num, shuffle=True, num_workers = 2, drop_last=True)

    datatensor, labeltensor = next(iter(test_loader))

    print('get random tensor for drawing',datatensor.size(),labeltensor.size())

    return datatensor,labeltensor


def get_mini_imagenet_mammalbird_data_loader():
    train_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_mammal_or_bird/train.csv')
    test_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_mammal_or_bird/test.csv')

    train_loader = DataLoader(dataset=train_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)

    print('train and test size: ', len(train_data), len(test_data))

    return train_loader, test_loader

def get_mini_imagenet_mammalbird_test_sample_tensor():
    test_data = mini_imagenet_Dataset('./data/mini-imagenet/contents_mammal_or_bird/test.csv')
    test_loader = DataLoader(dataset=test_data, batch_size = HP.sample_num, shuffle=True, num_workers = 2, drop_last=True)

    datatensor, labeltensor = next(iter(test_loader))

    print('get random tensor for drawing',datatensor.size(),labeltensor.size())

    return datatensor,labeltensor

def get_ut_zap50k_4_data_loader():
    train_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/train.csv',is_binary=False)
    test_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/test.csv',is_binary=False)

    train_loader = DataLoader(dataset=train_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)

    print('train and test size: ', len(train_data), len(test_data))

    return train_loader, test_loader

def get_ut_zap50k_4_test_sample_tensor():
    test_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/test.csv',is_binary=False)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.sample_num, shuffle=True, num_workers = 2, drop_last=True)

    datatensor, labeltensor = next(iter(test_loader))

    print('get random tensor for drawing',datatensor.size(),labeltensor.size())

    return datatensor,labeltensor


def get_ut_zap50k_2_data_loader():
    train_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/train.csv',is_binary=True)
    test_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/test.csv',is_binary=True)

    train_loader = DataLoader(dataset=train_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.batch_size, shuffle=True, num_workers = 2, drop_last=True)

    print('train and test size: ', len(train_data), len(test_data))

    return train_loader, test_loader

def get_ut_zap50k_2_test_sample_tensor():
    test_data = ut_zap50k_Dataset('./data/ut-zap50k-images/contents/test.csv',is_binary=True)
    test_loader = DataLoader(dataset=test_data, batch_size = HP.sample_num, shuffle=True, num_workers = 2, drop_last=True)

    datatensor, labeltensor = next(iter(test_loader))

    print('get random tensor for drawing',datatensor.size(),labeltensor.size())

    return datatensor,labeltensor

def get_cifar100_4_data_loader():
    training_set = torch.load('./data/re_cifar100/4_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/re_cifar100/4_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_CIFAR100_4_test_sample_tensor():
    training_set = torch.load('./data/re_cifar100/4_categories/test.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(10000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_cifar100_4_aug_data_loader():
    #train_data = CIFAR100Pair('./data/re_cifar100/7_categories/train.pt',transform=train_transform)
    #train_loader = DataLoader(train_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test_data = CIFAR100Pair('./data/re_cifar100/7_categories/test.pt',transform=test_transform)
    #test_loader = DataLoader(test_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Resize(size=(64,64))

    training_set = torch.load('./data/re_cifar100/4_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./data/re_cifar100/4_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    #print(training_data_tensors.size(), training_label_tensors.size())
    #print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_cifar100_7_aug_data_loader():
    #train_data = CIFAR100Pair('./data/re_cifar100/7_categories/train.pt',transform=train_transform)
    #train_loader = DataLoader(train_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test_data = CIFAR100Pair('./data/re_cifar100/7_categories/test.pt',transform=test_transform)
    #test_loader = DataLoader(test_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Resize(size=(64,64))

    training_set = torch.load('./data/re_cifar100/7_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    '''
    test_set = torch.load('./data/re_cifar100/7_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        img = to_pil(x)
        pos_1 = test_transform(img)
        Xs.append(pos_1)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    '''
    
    test_set = torch.load('./data/re_cifar100/7_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader


def get_cifar100_7_data_loader():
    training_set = torch.load('./data/re_cifar100/7_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    #normalize = transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])
    normalize = transforms.Resize(size=(64,64))
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/re_cifar100/7_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_CIFAR100_7_test_sample_tensor():
    training_set = torch.load('./data/re_cifar100/7_categories/test.pt')
    Xs = []
    Ys = []
    #normalize = transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])
    normalize = transforms.Resize(size=(64,64))
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        #x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(10000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_cifar100_3_data_loader():
    training_set = torch.load('./data/re_cifar100/3_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
        #print(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/re_cifar100/3_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_cifar100_3_aug_data_loader():
    #train_data = CIFAR100Pair('./data/re_cifar100/7_categories/train.pt',transform=train_transform)
    #train_loader = DataLoader(train_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test_data = CIFAR100Pair('./data/re_cifar100/7_categories/test.pt',transform=test_transform)
    #test_loader = DataLoader(test_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Resize(size=(64,64))

    training_set = torch.load('./data/re_cifar100/3_categories/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./data/re_cifar100/3_categories/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_CIFAR100_3_test_sample_tensor():
    training_set = torch.load('./data/re_cifar100/3_categories/test.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(10000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_cifar100_20_aug_data_loader():
    #train_data = CIFAR100Pair('./data/re_cifar100/7_categories/train.pt',transform=train_transform)
    #train_loader = DataLoader(train_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #test_data = CIFAR100Pair('./data/re_cifar100/7_categories/test.pt',transform=test_transform)
    #test_loader = DataLoader(test_data,batch_size=HP.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Resize(size=(64,64))

    training_set = torch.load('./data/cifar100_20/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./data/cifar100_20/test.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_CIFAR100_20_test_sample_tensor():
    training_set = torch.load('./data/cifar100_20/test.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(10000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_3_2_different_data_loader():
    training_set = torch.load('./data/3_2/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
        #print(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/3_2/test2.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_3_2_different_test_sample_tensor():
    training_set = torch.load('./data/3_2/test2.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(4000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_4_different_true_data_loader():
    training_set = torch.load('./data/4_3_2/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    normalize = transforms.Resize(size=(64,64))
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./data/4_3_2/test1.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_4_different_true_test_sample_tensor():
    training_set = torch.load('./data/4_3_2/test1.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(6000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_4_different_fake_data_loader():
    training_set = torch.load('./data/4_3_2/train.pt')
    print('length of training set: ', len(training_set))
    normalize = transforms.Resize(size=(64,64))
    to_pil = transforms.ToPILImage()
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)
    
    test_set = torch.load('./data/4_3_2/test2.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_4_different_fake_test_sample_tensor():
    training_set = torch.load('./data/4_3_2/test2.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(4000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_7_different_true_data_loader():
    training_set = torch.load('./data/7_3_2/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    normalize = transforms.Resize(size=(64,64))
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/7_3_2/test1.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_7_different_true_test_sample_tensor():
    training_set = torch.load('./data/7_3_2/test1.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(6000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def get_7_different_fake_data_loader():
    training_set = torch.load('./data/7_3_2/train.pt')
    print('length of training set: ', len(training_set))
    Xs = []
    Ys = []
    normalize = transforms.Resize(size=(64,64))
    to_pil = transforms.ToPILImage()
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        x = normalize(x)
        img = to_pil(x)
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        Xs.append(pos_1)
        Xs.append(pos_2)
        Ys.append(label)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)
    training_data_loader = DataLoader(dataset = TensorDataset(training_data_tensors, training_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)

    test_set = torch.load('./data/7_3_2/test2.pt')
    print('length of test set: ', len(test_set))
    Xs = []
    Ys = []
    for x, label in test_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    test_data_tensors = torch.stack(Xs)
    test_label_tensors = torch.tensor(Ys)
    test_data_loader = DataLoader(dataset = TensorDataset(test_data_tensors, test_label_tensors), batch_size = HP.batch_size, shuffle = True, num_workers = 2, drop_last=True)


    print(training_data_tensors.size(), training_label_tensors.size())
    print(test_data_tensors.size(), test_label_tensors.size())

    return training_data_loader, test_data_loader

def get_7_different_fake_test_sample_tensor():
    training_set = torch.load('./data/7_3_2/test2.pt')
    Xs = []
    Ys = []
    for x, label in training_set:
        #x = transforms.functional.to_tensor(img) # PIL to tensor
        Xs.append(x)
        Ys.append(label)
    training_data_tensors = torch.stack(Xs)
    training_label_tensors = torch.tensor(Ys)

    index = torch.LongTensor(random.sample(range(4000), HP.sample_num))
    training_data_tensors = torch.index_select(training_data_tensors, dim=0, index=index)
    training_label_tensors = torch.index_select(training_label_tensors, dim=0, index=index)

    print('get random tensor for drawing',training_data_tensors.size(),training_label_tensors.size())

    return training_data_tensors, training_label_tensors

def getData(dataset_name):
    if dataset_name == 'CIFAR10':
        return get_CIFAR10_dataloader()
    
    elif dataset_name == '101':
        return get_101_OC_data()

    elif dataset_name == 'CIFAR100':
        return get_CIFAR100_data_loader()

    elif dataset_name == 'FashionMNIST':
        return get_FashionMNIST_data_loader()

    elif dataset_name == 'VOC':
        return get_VOC_data_loader()

    elif dataset_name == 'MNIST_arc':
        return get_MNIST_arc_data_loader()

    elif dataset_name == 'MNIST_orientation':
        return get_MNIST_orientation_data_loader()

    elif dataset_name == 'mini-imagenet':
        return get_mini_imagenet_animal_data_loader()

    elif dataset_name == 'mini-imagenet-mb':
        return get_mini_imagenet_mammalbird_data_loader()

    elif dataset_name == 'ut-zap50k-4':
        return get_ut_zap50k_4_data_loader()

    elif dataset_name == 'ut-zap50k-2':
        return get_ut_zap50k_2_data_loader()

    elif dataset_name == 'CIFAR100-4':
        return get_cifar100_4_aug_data_loader()

    elif dataset_name == 'CIFAR100-7':
        return get_cifar100_7_aug_data_loader()

    elif dataset_name == 'CIFAR100-3':
        return get_cifar100_3_aug_data_loader()
    elif dataset_name == 'test':
        return get_test_data_loader()
    elif dataset_name == 'fake':
        return get_fake_data_loader()
    elif dataset_name == '3_2':
        return get_3_2_different_data_loader()
    elif dataset_name == 'CIFAR100-20':
        return get_cifar100_20_aug_data_loader()
    elif dataset_name == 'CIFAR100-4-TRUE':
        return get_4_different_true_data_loader()
    elif dataset_name == 'CIFAR100-4-FAKE':
        return get_4_different_fake_data_loader()
    elif dataset_name == 'CIFAR100-7-TRUE':
        return get_7_different_true_data_loader()
    elif dataset_name == 'CIFAR100-7-FAKE':
        return get_7_different_fake_data_loader()
    elif dataset_name == 'FMoW':
        return FMoW.get_data()
    elif dataset_name == 'iWildCam':
        return iWildCam.get_data()

    else:
        raise ValueError("No Such Dataset")

def get_sample_data(dataset_name):
    if dataset_name == 'CIFAR100':
        return get_CIFAR100_test_sample_tensor()
    elif dataset_name == 'mini-imagenet':
        return get_mini_imagenet_animal_test_sample_tensor()
    elif dataset_name == 'mini-imagenet-mb':
        return get_mini_imagenet_mammalbird_test_sample_tensor()
    elif dataset_name == 'ut-zap50k-4':
        return get_ut_zap50k_4_test_sample_tensor()
    elif dataset_name == 'ut-zap50k-2':
        return get_ut_zap50k_2_test_sample_tensor()
    elif dataset_name == 'CIFAR100-4':
        return get_CIFAR100_4_test_sample_tensor()
    elif dataset_name == 'CIFAR100-7':
        return get_CIFAR100_7_test_sample_tensor()
    elif dataset_name == 'CIFAR100-3':
        return get_CIFAR100_3_test_sample_tensor()
    elif dataset_name == 'test':
        return get_test_test_sample_tensor()
    elif dataset_name == 'fake':
        return get_fake_test_sample_tensor()
    elif dataset_name == '3_2':
        return get_3_2_different_test_sample_tensor()
    elif dataset_name == 'CIFAR100-20':
        return get_CIFAR100_20_test_sample_tensor()
    elif dataset_name == 'CIFAR100-4-TRUE':
        return get_4_different_true_test_sample_tensor()
    elif dataset_name == 'CIFAR100-4-FAKE':
        return get_4_different_fake_test_sample_tensor()
    elif dataset_name == 'CIFAR100-7-TRUE':
        return get_7_different_true_test_sample_tensor()
    elif dataset_name == 'CIFAR100-7-FAKE':
        return get_7_different_fake_test_sample_tensor()

    else:
        raise ValueError("No Such Dataset")