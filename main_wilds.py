import train
import hyperparameters as HP
from dataprocessor.datacomposer import getData
from model.TC2 import TransformerContrastive
from model.TC2 import get_backbone
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print('Dataset: ', HP.data_set)
backbone = HP.backbone
print(f'Backbone: {backbone}')

trainloader, testloader, grouper = getData(HP.data_set)

if HP.attention:
    net = TransformerContrastive()
else:
    #net = ResNet50(category_num=HP.cls_num)
    net = get_backbone(HP.backbone)
net = net.cuda()
#net = torch.nn.parallel.DataParallel(net, device_ids=[0])

print(net.named_children())


train.train_with_wilds(net,trainloader,testloader,HP.contrastive,grouper)