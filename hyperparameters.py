from asyncio import FastChildWatcher
import datetime
import time



batch_size = 64 # batch size in the training
learning_rate = 0.001 # learning rate in the training
epoch_num = 200 # max epoch number in the training

# params of CIA
dim_k = 256
dim_v = 128
n_heads = 8

# dataset
data_set = 'CIFAR100-4'

# backbone
backbone = 'ResNet50'

alpha = 0.5
lmd = 0.2

# model architecture
attention = True
contrastive  = True
G = True # projection from 2048 to 128, keep it True
TARGET = True




# output params
from utils import get_train_set_size
from utils import get_cls_num
train_set_size = get_train_set_size(data_set) 
cls_num = get_cls_num(data_set)

sample_num = 300

curr_time = datetime.datetime.now()


outname = f'[TARGET:{TARGET}]-[G:{G}]-[Attention:{attention}]-[Contrastive:{contrastive}]-[backbone:{backbone}]-[dataset:{data_set}]-[batch_size:{batch_size}]-[dim_k:{dim_k}]-[dim_v:{dim_v}]-[n_heads:{n_heads}]-[lr:{learning_rate}]-[alpha:{alpha}]-[lmd:{lmd}]-[time:{curr_time}]'

def get_outname():
    outname = f'[TARGET:{TARGET}]-[G:{G}]-[Attention:{attention}]-[Contrastive:{contrastive}]-[backbone:{backbone}]-[dataset:{data_set}]-[batch_size:{batch_size}]-[dim_k:{dim_k}]-[dim_v:{dim_v}]-[n_heads:{n_heads}]-[lr:{learning_rate}]-[alpha:{alpha}]-[lmd:{lmd}]-[time:{curr_time}]'
    return outname

#outname = f'[G:{G}]-[backbone:{backbone}]-[dataset:{data_set}]-[attention:{attention}]-[contrastive:{contrastive}]-time:{curr_time}'



