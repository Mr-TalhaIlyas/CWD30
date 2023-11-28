from configs.config import config
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def one_hot(index, classes):
    
    size = index.size()[:1] + (classes,) # jsut gets shape -> (P, C)
    view = index.size()[:1] + (1,) # get shapes -> (P, 1)
    
    # makes a tensor of size (P, C) and fills it with zeros
    mask = torch.Tensor(size).fill_(0).to('cuda' if torch.cuda.is_available() else 'cpu') # (P, C)
    # reshapes the targets/index to (P, 1)
    index = index.view(view) 
    ones = 1.

    return mask.scatter_(1, index, ones) # places ones at those indexes (in mask) indicated by values in targets 


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        '''
        only support ignore at 0.
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C => (P, C) 
        target = target.contiguous().view(-1) # (P,)
        
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, C) # (P, C)

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Entropy(nn.Module):
    def __init__(self, ita=2.0, charbonnier=False, reduce_dim=False):
        super(Entropy, self).__init__()
        self.ita = ita
        self.charbonnier = charbonnier
        self.reduce_dim = reduce_dim

    def forward(self, inputs):

        _, C, _, _ = inputs.size()
        P = F.softmax(inputs, dim=1)        # [B, C, H, W]
        logP = F.log_softmax(inputs, dim=1) # [B, C, H, W]
        PlogP = P * logP                    # [B, C, H, W]

        if self.reduce_dim:                 # for loss computation -> ent_loss
            ent = -1.0 * PlogP.sum(dim=1)   # [B, 1, H, W]
            ent = ent / np.log(C)
        else:                               # for trainig discriminator -> Prob2Ix
            ent = -1.0 * PlogP              # [B, C, H, W]
            ent = ent / np.log(C)   
        
        # robust entropy calculation
        if self.charbonnier:     
            # compute robust entropy
            ent = ent ** 2.0 + 1e-8
            ent = ent ** self.ita # [B, 1, H, W] -or- [B, C, H, W]

        if self.reduce_dim: 
            ent = ent.mean()

        return ent

# class_weight = torch.Tensor([0.500, 0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
#                             1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
#                             1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).to('cuda' if torch.cuda.is_available() else 'cpu')
# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=class_weight, reduction='none', ignore_index=-1):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
#                                    ignore_index=ignore_index)

#     def forward(self, inputs, targets):
#         # loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
#         loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
#         return loss.mean()
#         # return loss.mean(dim=2).mean(dim=1)


# def Entropy(x, ita=2.0, charbonnier=False):
    
#     _, C, _, _ = x.size()
#     P = F.softmax(x, dim=1)        # [B, C, H, W]
#     logP = F.log_softmax(x, dim=1) # [B, C, H, W]
#     PlogP = P * logP               # [B, C, H, W]
#     ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
#     ent = ent / torch.log(C)   
    
#     # robust entropy calculation
#     if charbonnier:     
#         # compute robust entropy
#         ent = ent ** 2.0 + 1e-8
#         ent = ent ** ita          # [B, 1, H, W]
           
#     return ent