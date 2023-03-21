import torch
import torch.nn as nn
from lib.pointnet import Pointnet2MSG
from lib.adaptor import PriorAdaptor

class deform_network(nn.Module):
    def __init__(self, n_cat=6):
        super(deform_network, self).__init__()
        self.n_cat = n_cat
        self.instance_geometry = Pointnet2MSG(0)
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.category_local = Pointnet2MSG(0)

        self.inst_self_attn = PriorAdaptor(emb_dims=64, n_heads=4)
        self.cat_self_attn = PriorAdaptor(emb_dims=64, n_heads=4)

        
        self.category_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.deformation[4].weight.data.normal_(0, 0.0001)

    
    def forward(self, points, cat_id, prior):

        bs = points.shape[0]
        nv = prior.size()[1]
        points = self.instance_geometry(points)
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        inst_attn_feats = self.inst_self_attn(points,points,points)
        inst_local = torch.cat((points, inst_attn_feats), dim=1)
        inst_global = self.instance_global(inst_local)   
        cat_points = self.category_local(prior) 
        cat_attn_feats = self.cat_self_attn(cat_points,cat_points,cat_points)
        cat_local = torch.cat((cat_points, cat_attn_feats), dim=1)
        cat_global = self.category_global(cat_local) 


        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()  
        deltas = torch.index_select(deltas, 0, index) 
        deltas = deltas.permute(0, 2, 1).contiguous()

        return deltas
