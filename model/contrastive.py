from turtle import forward
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, T=0.5):
        super().__init__()
        self.T = T
    def forward(self, x, y):
        representations = x
        label = y
        T = self.T
        n = label.shape[0]  # batch
        
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix.cuda()
        
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask = mask.cuda()

        
        mask_no_sim = torch.ones_like(mask) - mask

        
        mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )
        mask_dui_jiao_0 = mask_dui_jiao_0.cuda()

        
        similarity_matrix = torch.exp(similarity_matrix/T)

        
        similarity_matrix = similarity_matrix*mask_dui_jiao_0


        
        sim = mask*similarity_matrix


        
        no_sim = similarity_matrix - sim


        
        no_sim_sum = torch.sum(no_sim , dim=1)

        
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)


        
        #loss = torch.sum(mask_no_sim, loss, torch.eye(n, n ))
        loss = mask_no_sim + loss + (torch.eye(n, n )).cuda()


        
        loss = -torch.log(loss)  
        loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)

        return loss

#loss_func = ContrastiveLoss()


