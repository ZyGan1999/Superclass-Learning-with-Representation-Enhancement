import torch
import torch.nn.functional as F
import hyperparameters as HP
import torch.nn as nn

class TargetLoss(torch.nn.Module):
    def __init__(self, T=0.5):
        super().__init__()
        self.T = T
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 128, bias=True))
    def forward(self, x, y, target):
        '''
        x:tensor, with size (batch_size, emb_len)
        y:tensor, with size (batch_size), is the label of x
        target:tensor, with size (cls_num, emb_len)
        '''
        N = y.size(0)
        y = y.cuda()
        x = x.cuda()
        target = target.cuda()
        index = y.view(-1,1)
        index = index.cuda()
        one_hot = torch.zeros(y.size(0), target.size(0)).cuda().scatter_(1, index, 1) # (bz, cls) one-hot
        one_hot = one_hot.cuda()

        if not HP.attention:
            x = self.g(x)
        similarity_matrix = torch.mm(x, target.T) # (bz, cls)
        similarity_matrix = similarity_matrix.cuda()
        similarity_matrix = torch.div(similarity_matrix, self.T) # (bz, cls)
        similarity_matrix = torch.exp(similarity_matrix) # (bz, cls)
        
        target_similarity = torch.mul(similarity_matrix, one_hot) # (bz, cls) one hot with sim value
        target_similarity = target_similarity.cuda()
        target_similarity = torch.sum(target_similarity, dim=1) # (bz,1) target similarity of each data in the batch

        total_similarity = torch.sum(similarity_matrix, dim=1) # (bz,1)
        total_similarity = total_similarity.cuda()

        loss = torch.div(target_similarity, total_similarity) # (bz,1)
        loss = torch.log(loss)
        loss = torch.sum(loss,dim=0) # (1,1) the total loss of the batch

        loss = (-loss)/N

        return loss

#loss_func = ContrastiveLoss()


