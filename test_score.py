import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.utils import transform_points, make_non_exists_dir
from dataops.dataset import get_dataset_name
from Score_geo.utils.data import precompute_data_stack_mode
from Score_geo.utils.torch import to_cuda
from Score_geo.model import create_model
from Score_geo.config import make_cfg

def load_snapshot(model, snapshot):
    print('Loading from "{}".'.format(snapshot))
    state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
    assert 'model' in state_dict, 'No model can be loaded.'
    model.load_state_dict(state_dict['model'], strict=True)
    print('Model has been loaded.')

def normalize_scores(scores):
    mi = np.min(scores)
    ma = np.max(scores)
    uniform_scores = (scores-mi)/(ma-mi)
    return uniform_scores

class Tester():
    def __init__(self,cfg):
        self.cfg = cfg
        self.point_limit = 30000
        self.neighbor_limits = np.array([41, 36, 34, 15])
        model_cfg = make_cfg()
        self.model = create_model(model_cfg).cuda()
        load_snapshot(self.model,self.cfg.weight)
        self.model.eval()

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
        input_dict = precompute_data_stack_mode(points, lengths, num_stages=4, voxel_size=0.025, 
                                                radius=0.0625, neighbor_limits = self.neighbor_limits, point_num=128)
        collated_dict.update(input_dict)
        return(collated_dict)

    def run(self, datasets):
        pre_score = []
        ir_score = []
        labels = []
        save_dir = f'./data/auc_res'
        pc_dir = '/yxc/main/00_MINE/Scorer-GEO/data/3DMatch/data/test'
        make_non_exists_dir(save_dir)
        writer = open(f'{save_dir}/{self.cfg.dataset}_auc.log','w')
        writer.write(f'scene-id0-id1  label  score  ir \n')
        for scene,dataset in tqdm(datasets.items()):
            if scene=='wholesetname':continue
            trans_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/trans'
            for pair in tqdm(dataset.pair_ids):
                id0,id1=pair
                trans = dataset.get_transform(id0,id1)
                # trans = np.load(f'{trans_dir}/{id0}-{id1}.npz',allow_pickle=True)['trans']
                # ir = np.load(f'{trans_dir}/{id0}-{id1}.npz',allow_pickle=True)['ir']
                # label = np.load(f'{trans_dir}/{id0}-{id1}.npz',allow_pickle=True)['label']
                label = 1
                # labels.append(label)
                # ir_score.append(ir)
                pcd0 = dataset.get_pc_o3d(id0)
                pcd0 = pcd0.voxel_down_sample(0.025)
                pcd0 = np.array(pcd0.points)
                if pcd0.shape[0] > self.point_limit:
                    indices = np.random.permutation(pcd0.shape[0])[: self.point_limit]
                    pcd0 = pcd0[indices]
                pcd1 = dataset.get_pc_o3d(id1)
                pcd1 = pcd1.voxel_down_sample(0.025)
                pcd1 = np.array(pcd1.points)
                if pcd1.shape[0] > self.point_limit:
                    indices = np.random.permutation(pcd1.shape[0])[: self.point_limit]
                    pcd1 = pcd1[indices]
                pcd1 = transform_points(pcd1,trans)
                data_dict = {}
                data_dict['ref_points'] = pcd0.astype(np.float32)
                data_dict['src_points'] = pcd1.astype(np.float32)
                data_dict['ref_feats'] = np.ones((pcd0.shape[0], 1), dtype=np.float32)
                data_dict['src_feats'] = np.ones((pcd1.shape[0], 1), dtype=np.float32)

                collated_dict = self.dict_pre(data_dict)
                collated_dict = to_cuda(collated_dict)
                torch.cuda.synchronize()
                cls_logits = self.model(collated_dict)
                torch.cuda.synchronize()
                score = torch.sigmoid(cls_logits).detach().cpu().item()
                torch.cuda.empty_cache()
                pre_score.append(score)
                # writer.write(f'{scene}-{id0}-{id1}\t{label}\t{score}\t{ir}\n')
                writer.write(f'{scene}-{id0}-{id1}\t{label}\t{score}\n')
        # labels = np.array(labels)
        # np.save(f'{save_dir}/labels.npy',labels)
        # pre_score = np.array(pre_score)
        # np.save(f'{save_dir}/pre_score.npy',pre_score)
        # ir_score = np.array(ir_score)
        # ir_score = normalize_scores(ir_score)
        # np.save(f'{save_dir}/ir_score.npy',ir_score)
        # auc = roc_auc_score(labels,pre_score)
        # ir_auc = roc_auc_score(labels,ir_score)
        # print(ir_auc)
        # writer.write(f'auc: {auc}\nir_auc: {ir_auc}\n')
        # print(f'auc: {auc}\nir_auc: {ir_auc}\n')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='3dmatch',type=str,help='dataset name')
parser.add_argument('--origin_data_dir',type=str,default=f"./data/origin_data")
parser.add_argument('--output_cache_fn',type=str,default=f'./data/FCGF_Reg')
parser.add_argument('--weight',default='./Score_geo/weights/weight.pth.tar',type=str)
config = parser.parse_args()

datasets = get_dataset_name(config.dataset,config.origin_data_dir)
tester = Tester(config)
tester.run(datasets)

irs = np.load('/yxc/main/00_MINE/Scorer_test/FCGF-score/data/auc_res/ir_score.npy')
writer = open('./data/auc_res/ir_score.log','w')
for i in range(irs.shape[0]):
    ir = irs[i]
    writer.write(f'{ir}\n')
