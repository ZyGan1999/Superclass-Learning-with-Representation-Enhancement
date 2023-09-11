from importlib.metadata import metadata
from nis import cat
import hyperparameters as HP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import os

train_transform = transforms.Compose([
    #transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])])

test_transform = transforms.Compose([
    #transforms.Resize([128,128]), # 定义Resize类对象
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])])


class CIFAR100Pair(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.transform =transform
        self.imgs = torch.load(path)
        self.pil = transforms.ToPILImage()

    def __getitem__(self, index):
        img, target = self.imgs[index]
        img = self.pil(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            
            cat_pos = torch.cat([pos_1,pos_2],dim=0)
            cat_target = torch.cat([target,target],dim=0)
        return cat_pos, cat_target
    
    def __len__(self):
        return len(self.imgs)

class mini_imagenet_Dataset(Dataset):
    def __init__(self, filepath, transform=None, target_transform=None):
        contents = pd.read_csv(filepath)
        imgs = []
        for i in range(len(contents)):
            imgs.append(('./data/mini-imagenet/images/' + contents.iloc[i,0],contents.iloc[i,1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        img = img.resize((64, 64),Image.ANTIALIAS)
        img = transforms.functional.to_tensor(img) # PIL to tensor
        if self.transform is not None:
            img = self.transform(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)


class ut_zap50k_Dataset(Dataset):
    def __init__(self, filepath, transform=None, target_transform=None, is_binary=True):
        contents = pd.read_csv(filepath)
        imgs = []
        for i in range(len(contents)):
            if is_binary:
                imgs.append((contents.iloc[i,0],contents.iloc[i,2]))
            else:
                imgs.append((contents.iloc[i,0],contents.iloc[i,1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = fn.replace('./','./data/ut-zap50k-images/')
        img = Image.open(fn).convert('RGB')
        img = img.resize((100, 100),Image.ANTIALIAS)
        img = transforms.functional.to_tensor(img) # PIL to tensor
        if self.transform is not None:
            img = self.transform(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)



def single_channel_to_3_channel(ts):
    '''
    ts is a tensor with shape (len, h, w)
    output is a tensor with shape (len, 3, h, w)
    '''
    rtn_list = []
    for i in range(ts.shape[0]):
        tensor = ts[i]
        tri_tensor = torch.stack([tensor,tensor,tensor])
        rtn_list.append(tri_tensor)
    rtn = torch.stack(rtn_list)
    return rtn


def get_train_set_size(data_set):
    if data_set == 'CIFAR10':
        rtn = 50000
    elif data_set == '101':
        rtn = 7316
    elif data_set == 'CIFAR100':
        rtn = 50000
    elif data_set == 'FashionMNIST':
        rtn = 60000
    elif data_set == 'MNIST_arc':
        rtn = 60000
    elif data_set == 'MNIST_orientation':
        rtn = 60000
    elif data_set == 'VOC':
        rtn = 1905
    elif data_set == 'mini-imagenet':
        rtn = 50000
    elif data_set == 'mini-imagenet-mb':
        rtn = 4000
    elif data_set == 'ut-zap50k-4':
        rtn = 41687
    elif data_set == 'ut-zap50k-2':
        rtn = 41687
    elif data_set == 'CIFAR100-4':
        rtn = 50000
    elif data_set == 'CIFAR100-7':
        rtn = 50000
    elif data_set == 'CIFAR100-3':
        rtn = 50000
    elif data_set == 'test':
        rtn = 280
    elif data_set == 'fake':
        rtn = 2000
    elif data_set == '3_2':
        rtn = 30000
    elif data_set == 'CIFAR100-20':
        rtn = 50000
    elif data_set == 'CIFAR100-4-TRUE':
        rtn = 30000
    elif data_set == 'CIFAR100-4-FAKE':
        rtn = 30000
    elif data_set == 'CIFAR100-7-TRUE':
        rtn = 30000
    elif data_set == 'CIFAR100-7-FAKE':
        rtn = 30000
    elif data_set == 'FMoW':
        rtn = 76863
    elif data_set == 'iWildCam':
        rtn = 129809

    else:
        raise ValueError("No Such Dataset")

    print('Training Set Size:', rtn)

    return rtn

def get_cls_num(data_set):
    if data_set == 'CIFAR10':
        rtn = 2
    elif data_set == '101':
        rtn = 17
    elif data_set == 'CIFAR100':
        rtn = 3
    elif data_set == 'FashionMNIST':
        rtn = 2
    elif data_set == 'MNIST_arc':
        rtn = 2
    elif data_set == 'MNIST_orientation':
        rtn = 3
    elif data_set == 'VOC':
        rtn = 2
    elif data_set == 'mini-imagenet':
        rtn = 2
    elif data_set == 'mini-imagenet-mb':
        rtn = 2
    elif data_set == 'ut-zap50k-4':
        rtn = 4
    elif data_set == 'ut-zap50k-2':
        rtn = 2
    elif data_set == 'CIFAR100-4':
        rtn = 4
    elif data_set == 'CIFAR100-7':
        rtn = 7
    elif data_set == 'CIFAR100-3':
        rtn = 3
    elif data_set == 'test':
        rtn = 2
    elif data_set == 'fake':
        rtn = 2
    elif data_set == '3_2':
        rtn = 20
    elif data_set == 'CIFAR100-20':
        rtn = 20
    elif data_set == 'CIFAR100-4-TRUE':
        rtn = 4
    elif data_set == 'CIFAR100-4-FAKE':
        rtn = 4  
    elif data_set == 'CIFAR100-7-TRUE':
        rtn = 7
    elif data_set == 'CIFAR100-7-FAKE':
        rtn = 7
    elif data_set == 'FMoW':
        rtn = 6
    elif data_set == 'iWildCam':
        rtn = 182
    
    else:
        raise ValueError("No Such Dataset")
    
    print('Category Numbers: ', rtn)
    return rtn

def calc_accuracy(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        #不计算梯度，节省时间
        for step, (images,labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            _, outputs = net(images)
            numbers,predicted = torch.max(outputs,1)
            total +=labels.size(0)

            correct+=(predicted==labels).sum().item()
    return correct / total

def eval(net,testloader):
    correct = 0
    total = 0
    classnum = HP.cls_num
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.cuda(), targets.cuda()

            _,outputs = net(inputs)


            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask*tar_mask
            acc_num += acc_mask.sum(0)
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)

    rtn_recall_mean = recall.mean()
    rtn_precision = precision.mean()
    rtn_F1_mean = F1.mean()
    rtn_accuracy = accuracy
    

#精度调整
    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)
# 打印格式方便复制
    print('recall'," ".join('%s' % id for id in recall))
    print('precision'," ".join('%s' % id for id in precision))
    print('F1'," ".join('%s' % id for id in F1))
    print('accuracy',accuracy)


    return rtn_recall_mean.item(), rtn_precision.item(), rtn_F1_mean.item(), rtn_accuracy.item()


def eval_with_wilds(net,testloader,grouper):
    correct = 0
    total = 0
    classnum = HP.cls_num
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    with torch.no_grad():
        for batch_idx, (inputs, _, metadata) in enumerate(testloader):
            targets = grouper.metadata_to_group(metadata)

            #targets = _
            
            inputs, targets = inputs.cuda(), targets.cuda()

            _,outputs = net(inputs)


            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask*tar_mask
            acc_num += acc_mask.sum(0)
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)

    rtn_recall_mean = recall.mean()
    rtn_precision = precision.mean()
    rtn_F1_mean = F1.mean()
    rtn_accuracy = accuracy
    

#精度调整
    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)
# 打印格式方便复制
    print('recall'," ".join('%s' % id for id in recall))
    print('precision'," ".join('%s' % id for id in precision))
    print('F1'," ".join('%s' % id for id in F1))
    print('accuracy',accuracy)


    return rtn_recall_mean.item(), rtn_precision.item(), rtn_F1_mean.item(), rtn_accuracy.item()



def get_iter_dict(loader_dict):
    rtn = {}
    for key, value in loader_dict.items():
        rtn[key] = iter(value)
    return rtn



from collections.abc import Iterable

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def get_model_name():
    rtn = 'BASELINE'
    if HP.contrastive:
        rtn += '+CONTRA'
    if HP.attention:
        rtn += '+ATTENTION'

    return '||'+rtn+'||'

#tag_name = '[G]='+str(HP.G)+'[backbone]='+HP.backbone+'[dataset]='+HP.data_set+' - '+'[batch_size]='+str(HP.batch_size)+' - '+'[dim_k]='+str(HP.dim_k)+' - '+'[dim_v]='+str(HP.dim_v)+' - '+'[n_heads]='+str(HP.n_heads)+' - '+'[lr]='+str(HP.learning_rate) + ' - ' +'[alpha]='+str(HP.alpha)
tag_name = f'[TARGET:{HP.TARGET}]-[G:{HP.G}]-[backbone:{HP.backbone}]-[dataset:{HP.data_set}]-[batch_size:{HP.batch_size}]-[dim_k:{HP.dim_k}]-[dim_vL{HP.dim_v}]-[n_heads:{HP.n_heads}]-[lr:{HP.learning_rate}]-[alpha:{HP.alpha}]-[lmd:{HP.lmd}]'
#tag_name = HP.get_outname()

writer = SummaryWriter(comment = get_model_name()+tag_name)

def draw(X,Y,msg):
    '''
    X: tensor with shape (n, emb_len)
    Y: tensor with shape (n)
    msg: string for name of the output figure
    '''

    X = X.detach().numpy()
    Y = Y.detach().numpy()

    tsne = TSNE(n_components=2, learning_rate=200).fit_transform(X)
    plt.figure(figsize=(12, 12))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=Y, s=100)
    #plt.savefig('tsneimg/'+msg+'.png', dpi=120)
    if not os.path.exists(HP.outname):
        os.mkdir(HP.outname)
    plt.savefig(HP.outname+'/'+msg+'.png', dpi=120)
    plt.close()


