""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        

    def forward(self, data_list, label=None, infer=False):
        
        return MMLoss, MMlogit,tcp
    
    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

            


