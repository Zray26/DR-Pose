from email import message
from nis import cat
import os
import time
import argparse
import glob
from tqdm import tqdm
import _pickle as cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PoinTr.PoinTr_models.PoinTr import PoinTr
from lib.deform_net import deform_network
from DRPose_pipeline import Pipeline
from lib.align import estimateSimilarityTransform
from data.neighbor import collate_fn
import cv2
from easydict import EasyDict as edict
import yaml
from lib.align import estimateSimilarityTransform
from lib.utils_regis import load_depth, get_bbox, compute_mAP
import os
cwd = os.getcwd()
print(cwd)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='real_test', help='real_test or camera_val')
parser.add_argument('--data_dir', type=str, default='dataset/NOCS', help='data directory')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--deformation_model', type=str, default='weights/deformation/deformation_model.pth', help='resume from saved model')
parser.add_argument('--registration_model', type=str, default='weights/registration/registration_model.pth', help='resume from saved model')
parser.add_argument('--completion_model', type=str, default='weights/completion/completion_model.pth', help='resume from saved model')
parser.add_argument('--config_dir',type=str,default='configs/config.yaml')
opt = parser.parse_args()
deformation_model = os.path.join(cwd,opt.deformation_model)
registration_model = os.path.join(cwd,opt.registration_model)
completion_model = os.path.join(cwd,opt.completion_model)
config_dir = os.path.join(cwd,opt.config_dir)
data_dir = os.path.join(cwd,'dataset/NOCS')
opt.decay_epoch = [0, 5, 10]
opt.decay_rate = [1.0, 0.6, 0.3]
opt.corr_wt = 1.0
opt.cd_wt = 5.0
opt.entropy_wt = 0.0001
opt.deform_wt = 0.01
if opt.dataset == 'camera_val':
    result_dir = os.path.join(cwd,'Camera_results')
else:
    result_dir = os.path.join(cwd,'Real_results')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

prior_dir = os.path.join(cwd,'assets/mean_points_emb.npy')
mean_shapes = np.load(prior_dir)

assert opt.dataset in ['camera_val', 'real_test']

if opt.dataset == 'camera_val':
    file_path = 'CAMERA/val_list.txt'
    result_dir = os.path.join(cwd,'Camera_results')
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = os.path.join(cwd,'Real_results')
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0

def detect():
    with open(config_dir) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)
    kpfcn_backbone = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary'
]
    config.kpfcn_config.architecture = kpfcn_backbone

    completion_net = PoinTr()
    completion_net = completion_net.cuda()
    completion_net = nn.DataParallel(completion_net)
    completion_net.load_state_dict(torch.load(completion_model))
    completion_net.eval()

    deform_net = deform_network()
    deform_net.cuda()
    deform_net = nn.DataParallel(deform_net)
    deform_net.load_state_dict(torch.load(deformation_model))

    registration_net = Pipeline(config)
    registration_net = registration_net.cuda()
    registration_net.load_state_dict(torch.load(registration_model,map_location='cuda:0'))

    deform_net.train()
    registration_net.eval()

    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(data_dir, file_path))]
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    for path in tqdm(img_list):
        max_points=6000
        img_path = os.path.join(data_dir, path)
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)
        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]
        img_path_parsing = img_path.split('/')
        mrcnn_dir = os.path.join(data_dir,'results/mrcnn_results')
        if opt.dataset == 'camera_val':
            mrcnn_path = os.path.join(mrcnn_dir, 'val', 'results_{}_{}_{}.pkl'.format(
                opt.dataset.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        else:
            mrcnn_path = os.path.join(mrcnn_dir, 'real_test', 'results_{}_{}_{}.pkl'.format(
                opt.dataset.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)

        f_points, f_choose, f_catId, f_prior, f_nocs,f_fullpoints = [], [], [], [], [], []
        valid_inst = []
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            prior = mean_shapes[cat_id]
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]


            if len(choose) > max_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:max_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                pass
            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            
            
            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            full_points = np.concatenate((pt0, pt1, pt2), axis=1)
            nocs = coord[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5

            if len(choose) > opt.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')

            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            points = np.concatenate((pt0, pt1, pt2), axis=1)

            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)

            # concatenate instances
            f_points.append(points)
            f_choose.append(choose)
            f_catId.append(cat_id)
            f_prior.append(prior)
            f_nocs.append(nocs)
            f_fullpoints.append(full_points)
        if len(valid_inst):
            with torch.no_grad():
                f_points = torch.cuda.FloatTensor(f_points)
                f_rgb = torch.cuda.LongTensor(f_catId.copy())
                f_choose = torch.cuda.LongTensor(f_choose)
                f_catId = torch.cuda.LongTensor(f_catId)
                f_prior = torch.cuda.FloatTensor(f_prior)
                torch.cuda.synchronize()
                t_now = time.time()
                centers = torch.unsqueeze(f_points.mean(1),1)
                norm_points = f_points-centers
                dists = torch.sqrt((norm_points ** 2).sum(2))
                dists = torch.max(dists,1).values * 2
                dists = torch.unsqueeze(dists,1)
                dists = torch.unsqueeze(dists,1)
                norm_points/=dists
                ret = completion_net(norm_points)
                deltas = torch.empty([f_points.shape[0],1024,3]).cuda()
                for z in range(len(valid_inst)):
                    if f_catId[z] in [4, 5]:
                        completed_points = torch.unsqueeze(f_points[z],0)
                    else:
                        # completed_points = points
                        if opt.dataset == 'camera_val':
                            completed_points = torch.unsqueeze(ret[1][z],0)
                        else:
                            completed_points = torch.unsqueeze(ret[1][z][:-1024],0)
                        completed_points = (completed_points *dists[z]) +centers[z]
                    delta = deform_net(completed_points, torch.unsqueeze(f_catId[z],0), torch.unsqueeze(f_prior[z],0))
                    deltas[z] = delta
                inst_shape = deltas+f_prior
                inst_shape = inst_shape.detach().cpu()
            # inst_shape = prior + deltas

            data_list = []
            for t in range(len(valid_inst)):
                s_t = []
                if t == 0:
                    nocs = f_nocs[t]
                    full_points = f_fullpoints[0]
                else:
                    nocs = np.concatenate((nocs,f_nocs[t]),axis=0)
                    full_points = np.concatenate((full_points,f_fullpoints[t]),axis=0)
                s_t.append(f_fullpoints[t]) #full_points
                s_t.append(np.array(inst_shape[t])) #should be deformed prior
                data_list.append(s_t)

            data_out = collate_fn(data_list,config)
            del data_list
            batch_len = data_out['stack_lengths']
            len_2 = batch_len[-2]
            for k, v in data_out.items():
                if type(v) == list:
                    data_out [k] = [item.cuda() for item in v] # send everything to gpu
                elif type(v) in [ dict, float, type(None), np.ndarray]:
                    pass
                else:
                    data_out [k] = v.cuda()
            src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, scale_mat = registration_net(data_out)
            pool_id = data_out['pools'][1][:,0]
            id_accu = 0
            prior_count = 0
            f_coords = []
            f_selected_points = []
            f_nocs = []
            for j in range(len(valid_inst)):
                src_len = len_2[j*2]
                tgt_len = len_2[j*2+1]
                scales = scale_mat[j,0,:src_len]
                scales = torch.clamp(scales,-0.4, 0.4)
                scales = torch.unsqueeze(scales,1)
                feat_x = src_feats[j, :, :src_len]
                feat_y = tgt_feats[j, :, :tgt_len]
                assign = torch.matmul(feat_x.T,feat_y)
                soft_assign = F.softmax(assign,dim=1)
                deformed_prior = t_pcd[j,:tgt_len,:]
                coords = torch.matmul(soft_assign,deformed_prior)
                coords = coords * (1+ scales)
                ind_layer1 = pool_id[id_accu:(id_accu+len_2[j*2])].detach().cpu()
                ind_layer0 = data_out['pools'][0][ind_layer1][:,0].detach().cpu()
                raw_nocs_ind = ind_layer0 - prior_count * 1024
                non_max_ind = raw_nocs_ind !=len(data_out['points'][0])
                nocs_ind = raw_nocs_ind[non_max_ind]
                prior_count += 1
                selected_points = full_points[nocs_ind]
                coords = coords[non_max_ind]
                f_coords.append(coords)
                f_selected_points.append(selected_points)
                id_accu = id_accu + len_2[j*2] + len_2[j*2 + 1]
            torch.cuda.synchronize()
            t_inference += (time.time() - t_now)
            t_now = time.time()
            for q in range(len(valid_inst)):
                inst_idx = valid_inst[q]
                nocs_coords = f_coords[q]
                f_size[inst_idx] = 2 * np.amax(np.abs(inst_shape[q].numpy()), axis=0)
                points = f_selected_points[q]
                _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords.detach().cpu(), points)
                if pred_sRT is None:
                    pred_sRT = np.identity(4, dtype=float)
                f_sRT[inst_idx] = pred_sRT
            t_umeyama += (time.time() - t_now)
            img_count += 1
            inst_count += len(valid_inst)

        # save results
        result = {}
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['class_ids']
        result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_scores'] = mrcnn_result['scores']
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size

        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)
    # write statistics
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference/img_count))
    messages.append("Umeyama time: {:06f}  Average: {:06f}/image".format(t_umeyama, t_umeyama/img_count))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    messages = []
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()

if __name__ == '__main__':
    print('Detecting ...')
    detect()
    print('Evaluating ...')
    evaluate()