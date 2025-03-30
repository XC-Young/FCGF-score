import torch
import pickle
import numpy as np
import argparse
import open3d as o3d
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.utils import transform_points, make_non_exists_dir
from utils.r_eval import compute_R_diff
from dataops.dataset import get_dataset_name
from Score_geo.utils.data import precompute_data_stack_mode
from Score_geo.utils.torch import to_cuda
from Score_geo.model import create_geo_model,create_score_model
from Score_geo.config import make_cfg

def load_snapshot(geo_model,score_model, snapshot):
    print('Loading from "{}".'.format(snapshot))
    state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
    assert 'model' in state_dict, 'No model can be loaded.'
    geo_params,score_params = {},{}
    for key, value in state_dict['model'].items():
        if key.startswith('backbone.') or key.startswith('transformer.'):
            geo_params[key] = value
        elif key.startswith('cls_head.'):
            score_params[key] = value
    geo_model.load_state_dict(geo_params, strict=True)
    score_model.load_state_dict(score_params, strict=True)
    print('Model has been loaded.')

def normalize_scores(scores):
    mi = np.min(scores)
    ma = np.max(scores)
    uniform_scores = (scores-mi)/(ma-mi)
    return uniform_scores

def save_pkl_scannet(datasets):
    pkl_list = []
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        for pair in tqdm(dataset.pair_ids):
            id0,id1 = pair
            gt = dataset.get_transform(id0,id1)
            gt = np.concatenate([gt,[[0.0,0.0,0.0,1.0]]],axis=0)
            rotation = gt[0:3:,0:3]
            translation = gt[0:3,3]
            # cal overlap
            ply0 = dataset.get_pc_o3d(id0)
            ply1 = dataset.get_pc_o3d(id1)
            voxel_size = 0.025
            ply0 = ply0.voxel_down_sample(voxel_size)
            ply1 = ply1.voxel_down_sample(voxel_size)

            ply1.transform(gt)
            ply0_tree = o3d.geometry.KDTreeFlann(ply1)
            ply1_tree = o3d.geometry.KDTreeFlann(ply0)
            threshold = 0.1

            overlap_count0 = 0
            for point in ply0.points:
                [_, idx, dists] = ply0_tree.search_knn_vector_3d(point, 1)
                if len(dists) > 0 and np.sqrt(dists[0]) < threshold:
                    overlap_count0 += 1
            overlap_ratio0 = overlap_count0 / len(ply0.points)

            overlap_count1 = 0
            for point in ply1.points:
                [_, idx, dists] = ply1_tree.search_knn_vector_3d(point, 1)
                if len(dists) > 0 and np.sqrt(dists[0]) < threshold:
                    overlap_count1 += 1
            overlap_ratio1 = overlap_count1 / len(ply1.points)
            overlap = (overlap_ratio0 + overlap_ratio1)/2

            metadata = {
                'overlap':overlap,
                'pcd0':f'{scene}/PointCloud/cloud_bin_{id0}.ply',
                'pcd1':f'{scene}/PointCloud/cloud_bin_{id1}.ply',
                'rotation':rotation,
                'translation':translation,
                'scene_name':scene,
                'frag_id0':id0,
                'frag_id1':id1,
            }
            pkl_list.append(metadata)

    with open('./scannet.pkl','wb') as f:
        pickle.dump(pkl_list,f)

class Tester():
    def __init__(self,cfg):
        self.cfg = cfg
        self.point_limit = 30000
        self.neighbor_limits = np.array([41, 35, 34, 15])
        model_cfg = make_cfg()
        self.geo_model = create_geo_model(model_cfg).cuda()
        self.score_model = create_score_model(model_cfg).cuda()
        load_snapshot(self.geo_model,self.score_model,self.cfg.weight)
        self.geo_model.eval()
        self.score_model.eval()

    def dict_pre(self,data_dict):
        collated_dict = {}
        # array to tensor
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

        # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
        feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
        points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
        lengths = torch.LongTensor([points.shape[0] for points in points_list])
        points = torch.cat(points_list, dim=0)
        # remove wrapping brackets
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]
        collated_dict['features'] = feats
        input_dict = precompute_data_stack_mode(points, lengths, num_stages=4, voxel_size=self.cfg.voxel, 
                                                radius=0.0625, neighbor_limits = self.neighbor_limits, point_num=128)
        collated_dict.update(input_dict)
        return(collated_dict)

    def run(self, datasets):
        pre_score = []
        irs,mses,maes,cds = [],[],[],[]
        labels = []
        save_dir = f'./data/auc_res'
        pc_dir = '/yxc/main/00_MINE/Scorer-GEO/data/3DMatch/data/test'
        make_non_exists_dir(save_dir)
        writer = open(f'{save_dir}/{self.cfg.dataset}_auc.log','w')
        writer.write(f'scene-id0-id1  label  score  ir \n')
        for scene,dataset in tqdm(datasets.items()):
            if scene=='wholesetname':continue
            trans_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/ir_top_trans'
            for pair in tqdm(dataset.pair_ids):
                id0,id1=pair
                # for gt score
                # trans = dataset.get_transform(id0,id1)
                # label = 1
                top_trans = np.load(f'{trans_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans'][0]
                trans = top_trans['trans']
                ir = top_trans['inlier_ratio']
                mse = top_trans['mse']
                mae = top_trans['mae']
                cd = top_trans['cd']
                gt = dataset.get_transform(id0,id1)
                if trans.shape[0] == 3:
                    trans = np.concatenate([trans,[[0.0,0.0,0.0,1.0]]],axis=0)
                rte = np.linalg.norm(gt[0:3,3]-trans[0:3,3])
                rre = compute_R_diff(gt[0:3:,0:3:],trans[0:3:,0:3:])
                if rre <= 15 and rte <= 0.3:
                    label = 1
                else:
                    label = 0
                labels.append(label)
                irs.append(ir)
                mses.append(mse)
                maes.append(mae)
                cds.append(cd)
                pcd0 = dataset.get_pc_o3d(id0)
                pcd0 = pcd0.voxel_down_sample(self.cfg.voxel)
                pcd0 = np.array(pcd0.points)
                if pcd0.shape[0] > self.point_limit:
                    indices = np.random.permutation(pcd0.shape[0])[: self.point_limit]
                    pcd0 = pcd0[indices]
                pcd1 = dataset.get_pc_o3d(id1)
                pcd1 = pcd1.voxel_down_sample(self.cfg.voxel)
                pcd1 = np.array(pcd1.points)
                if pcd1.shape[0] > self.point_limit:
                    indices = np.random.permutation(pcd1.shape[0])[: self.point_limit]
                    pcd1 = pcd1[indices]
                data_dict = {}
                data_dict['ref_points'] = pcd0.astype(np.float32)
                data_dict['src_points'] = pcd1.astype(np.float32)
                data_dict['ref_feats'] = np.ones((pcd0.shape[0], 1), dtype=np.float32)
                data_dict['src_feats'] = np.ones((pcd1.shape[0], 1), dtype=np.float32)

                collated_dict = self.dict_pre(data_dict)
                collated_dict = to_cuda(collated_dict)
                ref_feats_c_norm,src_feats_c_norm = self.geo_model(collated_dict)
                torch.cuda.synchronize()
                trans_g = to_cuda(torch.from_numpy(trans))        
                cls_logits = self.score_model(collated_dict,ref_feats_c_norm,src_feats_c_norm,trans_g)
                torch.cuda.synchronize()
                score = torch.sigmoid(cls_logits).detach().cpu().item()
                torch.cuda.empty_cache()
                pre_score.append(score)
                writer.write(f'{scene}-{id0}-{id1}\t{label}\t{score}\t{ir}\t{mse}\t{mae}\t{cd}\n')
        labels = np.array(labels)
        np.save(f'{save_dir}/{self.cfg.dataset}_labels.npy',labels)
        pre_score = np.array(pre_score)
        np.save(f'{save_dir}/{self.cfg.dataset}_score.npy',pre_score)
        irs,mses,maes,cds = np.array(irs),np.array(mses),np.array(maes),np.array(cds)
        ir_norm = normalize_scores(irs)
        mse_norm = 1 - normalize_scores(mses)
        mae_norm = 1 - normalize_scores(maes)
        cd_norm = 1 - normalize_scores(cds)
        np.save(f'{save_dir}/{self.cfg.dataset}_ir.npy',ir_norm)
        np.save(f'{save_dir}/{self.cfg.dataset}_mse.npy',mse_norm)
        np.save(f'{save_dir}/{self.cfg.dataset}_mae.npy',mae_norm)
        np.save(f'{save_dir}/{self.cfg.dataset}_cd.npy',cd_norm)

        auc = roc_auc_score(labels,pre_score)
        ir_auc = roc_auc_score(labels,ir_norm)
        mse_auc = roc_auc_score(labels,mse_norm)
        mae_auc = roc_auc_score(labels,mae_norm)
        cd_auc = roc_auc_score(labels,cd_norm)
        writer.write(f'auc: {auc}\nir_auc: {ir_auc}\nmse_auc: {mse_auc}\nmae_auc: {mae_auc}\ncd_auc: {cd_auc}\n')
        print(f'auc: {auc}\nir_auc: {ir_auc}\nmse_auc: {mse_auc}\nmae_auc: {mae_auc}\ncd_auc: {cd_auc}\n')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='3dmatch',type=str,help='dataset name')
parser.add_argument('--origin_data_dir',type=str,default=f"./data/origin_data")
parser.add_argument('--output_cache_fn',type=str,default=f'./data/FCGF_Reg')
parser.add_argument('--weight',default='./Score_geo/weights/epoch-2.pth.tar',type=str)
parser.add_argument('--voxel',default=0.025,type=float)
config = parser.parse_args()

datasets = get_dataset_name(config.dataset,config.origin_data_dir)
tester = Tester(config)
tester.run(datasets)

""" irs = np.load('/yxc/main/00_MINE/Scorer_test/FCGF-score/data/auc_res/ir_score.npy')
writer = open('./data/auc_res/ir_score.log','w')
for i in range(irs.shape[0]):
    ir = irs[i]
    writer.write(f'{ir}\n') """
