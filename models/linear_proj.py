from turtle import forward
import torch
import torch.nn as nn

class Proj_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_s = nn.Conv1d(528,528,1)
        self.mlp_t = nn.Conv1d(528,528,1)

    def forward(self,src,tgt):
        return self.mlp_s(src),self.mlp_t(tgt)
