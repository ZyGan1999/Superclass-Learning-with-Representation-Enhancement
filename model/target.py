from math import sqrt
from webbrowser import get
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F

class TargetGenerator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.FC = nn.Sequential(nn.Linear(self.dim_in, self.dim_in, bias=False),
                               nn.Tanh(),
                               nn.Linear(self.dim_in, self.dim_in, bias=False),
                               nn.Tanh(),
                               nn.Linear(self.dim_in, self.dim_out, bias=False))
        
        
    def forward(self, x):
        out = self.FC(x)

        # L2 norm
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        
        return out


class SeperateLoss(torch.nn.Module):
    def __init__(self, C, T=0.5):
        super().__init__()
        self.T = T
        self.C = C
    def forward(self, x):
        sim = torch.mm(x, x.T) # nxn
        sim = torch.div(sim, self.T) # nxn
        sim = torch.exp(sim) # nxn
        sim = torch.sum(sim, dim=1) # nx1
        sim = torch.log(sim) # nx1
        sim = torch.sum(sim, dim=0) # 1x1

        loss = sim

        loss /= self.C

        return loss

def inc_label_tensor(num):
    '''
    to return a tensor = [0,1,2,...,num-1]
    '''
    return torch.tensor([i for i in range(num)])

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
    plt.scatter(tsne[:, 0], tsne[:, 1], c=Y)
    #plt.savefig('tsneimg/'+msg+'.png', dpi=120)
    if not os.path.exists('test'):
        os.mkdir('test')
    plt.savefig('test'+'/'+msg+'.png', dpi=120)
    plt.close()


def get_target(num, dim):
    '''
    given num and dim of the targets, generate the target
    1. randomly generate #num targets, which has length of #dim, and is L2 normalized
    2. to train the generator(a simple MLP), based on the loss function which can seperate the targets
    3. return the target(size = num x dim), it is a tensor
    '''

    Y = inc_label_tensor(num=num)

    net = TargetGenerator(dim_in=dim, dim_out=dim)

    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

    loss_func = SeperateLoss(C=num)

    bestmodel = net

    t = torch.rand(num, dim)
    t = net(t)
    min_loss = float(loss_func(t))
    
    for epoch in range(10):
        target = torch.rand(num, dim)
        target = net(target)
        loss = loss_func(target)

        if loss < min_loss:
            min_loss = float(loss)
            bestmodel = net

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #draw(X=target,Y=Y,msg=f'epoch={epoch}')
        #print(loss)
    target = torch.rand(num, dim)
    target = bestmodel(target)

    return target


#get_target(num=7,dim=2048)


class Target():
    def __init__(self) -> None:
        pass

    def generate_target(self,num,dim):
        self.target = get_target(num,dim)
        torch.save(self.target,'target.pt')

    def get_target(self):
        rtn = torch.load('target.pt')
        return rtn