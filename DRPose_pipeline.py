import yaml
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from models.backbone import KPFCN
from models.linear_proj import Proj_layer
from models.transformer import RepositioningTransformer
def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)



class Pipeline(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.backbone = KPFCN(config['kpfcn_config'])
        self.mlp = Proj_layer()
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])
        self.category_global = nn.Sequential(
            nn.Conv1d(528, 528, 1),
            nn.ReLU(),
            nn.Conv1d(528, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.inst_global = nn.Sequential(
            nn.Conv1d(528, 528, 1),
            nn.ReLU(),
            nn.Conv1d(528, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.scale_net = nn.Sequential(
            nn.Conv1d(2576, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1),
        )
        
    def forward(self,data):
        coarse_feats = self.backbone(data)
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = split_feats (coarse_feats, data)
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data)
        src_feats = src_feats.permute(0,2,1)
        tgt_feats = tgt_feats.permute(0,2,1)
        nv = src_feats.shape[-1]
        bs = len(src_mask)
        scale_mat = torch.empty([bs,1,nv]).cuda()
        for i in range(bs):
            src_len = src_mask[i].sum()
            tgt_len = tgt_mask[i].sum()
            src_feats_single = torch.unsqueeze(src_feats[i][:,:src_len],0)
            tgt_feats_single = torch.unsqueeze(tgt_feats[i][:,:tgt_len],0)
            inst_global = self.inst_global(src_feats_single)
            cat_global = self.category_global(tgt_feats_single)
            scale_feat = torch.cat((torch.unsqueeze(src_feats[i],0), inst_global.repeat(1, 1, nv), cat_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
            scale_mat[i] = self.sigmoid(self.scale_net(scale_feat)) -0.5
        src_feats, tgt_feats = self.mlp(src_feats,tgt_feats)
        src_feats/=(src_feats.shape[1])**.5
        tgt_feats/=(tgt_feats.shape[1])**.5
        return src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, scale_mat

    def sigmoid(self, z):
        return 1/(1+(-z).exp())


def split_feats(geo_feats, data):

    pcd = data['points'][-2]
    src_mask = data['src_mask']
    tgt_mask = data['tgt_mask']
    src_ind_coarse_split = data[ 'src_ind_coarse_split']
    tgt_ind_coarse_split = data['tgt_ind_coarse_split']
    src_ind_coarse = data['src_ind_coarse']
    tgt_ind_coarse = data['tgt_ind_coarse']

    b_size, src_pts_max = src_mask.shape
    tgt_pts_max = tgt_mask.shape[1]

    src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
    tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
    src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
    tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

    src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
    tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
    src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
    tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

    return src_feats.view( b_size , src_pts_max , -1), \
            tgt_feats.view( b_size , tgt_pts_max , -1), \
            src_pcd.view( b_size , src_pts_max , -1), \
            tgt_pcd.view( b_size , tgt_pts_max , -1), \
            src_mask, \
            tgt_mask