import imp
from operator import imod
import train
import hyperparameters as HP
from dataprocessor.datacomposer import get_CIFAR10_re_dataset
from dataprocessor.datacomposer import get_101_OC_data
from dataprocessor.datacomposer import get_101_data_split_by_macro_label
from dataprocessor.datacomposer import getData
from dataprocessor.datacomposer import get_CIFAR100_data_loader
from model.TC2 import TransformerContrastive
from model.TC2 import get_backbone
from model.resnet50 import ResNet50
from model.CNN import CNN_NET
import torch
import os
from model.target import get_target
from model.TC2 import get_emb_len

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#label_to_idx, train_data_loader_dict, test_loader = get_101_data_split_by_macro_label(is_enumerate=False)
#idx_to_label = dict(zip(label_to_idx.values(), label_to_idx.keys()))

print('Dataset: ', HP.data_set)
backbone = HP.backbone
print(f'Backbone: {backbone}')

trainloader, testloader = getData(HP.data_set)

if HP.attention:
    net = TransformerContrastive()
else:
    #net = ResNet50(category_num=HP.cls_num)
    net = get_backbone(HP.backbone)
net = net.cuda()
net = torch.nn.parallel.DataParallel(net, device_ids=[0,1])

#print(net.named_children())


#train.train_by_gathering_same_label_data_in_one_batch(net, label_to_idx, train_data_loader_dict, test_loader)
#train.train_by_allocate_different_label_in_one_batch(net, label_to_idx, train_data_loader_dict, test_loader)
'''
if HP.contrastive:
    train.train_con(net,trainloader,testloader)
else:
    train.train_raw(net,trainloader,testloader)
'''


train.train(net,trainloader,testloader,HP.contrastive)