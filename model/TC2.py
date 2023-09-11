from math import sqrt

import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import hyperparameters as HP
from torchvision.models.resnet import resnet50

from model.resnetxx import ResNet50
from model.CNN import CNN_NET
from model.resnetxx import ResNet18
from model.resnetxx import ResNet34
from model.resnetxx import ResNet101
from model.resnetxx import ResNet152


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        #print(x.device)

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att



def get_backbone(backbone):
    if backbone == 'ResNet50':
        return ResNet50(category_num=HP.cls_num)
    elif backbone == 'ResNet18':
        return ResNet18(category_num=HP.cls_num)
    elif backbone == 'ResNet34':
        return ResNet34(category_num=HP.cls_num)
    elif backbone == 'ResNet101':
        return ResNet101(category_num=HP.cls_num)
    elif backbone == 'ResNet152':
        return ResNet152(category_num=HP.cls_num)
    else:
        raise ValueError("No Such Backbone")

def get_emb_len(backbone):
    if backbone == 'ResNet50':
        return 2048
    elif backbone == 'ResNet18':
        return 512
    elif backbone == 'ResNet34':
        return 512
    elif backbone == 'ResNet101':
        return 2048
    elif backbone == 'ResNet152':
        return 2048
    else:
        raise ValueError("No Such Backbone")


class TransformerContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        #self.slf_attn = MultiHeadSelfAttention(dim_in = get_emb_len(HP.backbone), dim_k=HP.dim_k, dim_v=HP.dim_v, num_heads=HP.n_heads) # transformer module
        self.slf_attn = MultiHeadSelfAttention(dim_in = 128, dim_k=HP.dim_k, dim_v=HP.dim_v, num_heads=HP.n_heads) # transformer module
        #self.slf_embed = ResNet50(category_num=HP.cls_num) # embedding module
        self.slf_embed = get_backbone(HP.backbone)
        # contrastive learning module

        #self.linear = nn.Linear(HP.dim_v, HP.cls_num)

        self.linear = nn.Sequential(nn.Linear(HP.dim_v, HP.dim_v),
                                    nn.Linear(HP.dim_v, 64),
                                    nn.Linear(64,32),
                                    nn.Linear(32,HP.cls_num))

        '''
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 128, bias=True),
                               nn.Linear(128, HP.cls_num))
        '''
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 128, bias=True))
        self.prd = nn.Linear(128, HP.cls_num)
        

    def forward(self, x):
        embedded_data, _ = self.slf_embed(x) # get the embedding of the data of a batch by resnet50
        #embedded_data = self.slf_embed(x)

        if HP.G:
            embedded_data = self.g(embedded_data) # projection head

        embedded_data = torch.reshape(embedded_data,(1,embedded_data.size(0),embedded_data.size(1))) # (batchsize, emb_len) -> (1(batchsize), batchsize(n), emb_len)
        attn = self.slf_attn(embedded_data)
        attn = torch.reshape(attn, (attn.size(1), attn.size(2))) # (1, batchsize(n), dim_v) -> (batchsize(n), dim_v)
        
        # L2 Regulization
        norm = torch.norm(attn,2,1,True)
        attn = torch.div(attn, norm)

        '''
        if HP.G:
            output = self.g(attn)
            #attn = self.g(attn)
            #output = self.prd(attn)
        else:
            output = self.linear(attn)
        '''

        output = self.linear(attn)

        return attn, output
