import numpy as np
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time,os
import open3d as o3d
from utils.utils import transform_points, make_non_exists_dir
from utils.r_eval import compute_R_diff
from dataops.dataset import get_dataset_name
from utils.knn_search import knn_module
import utils.RR_cal as RR_cal
from Score_geo.utils.data import precompute_data_stack_mode
from Score_geo.utils.torch import to_cuda
from Score_geo.model import create_geo_model,create_score_model
from Score_geo.config import make_cfg

# match the feature
class matcher():
    def __init__(self,cfg):
        self.cfg = cfg
        self.KNN = knn_module.KNN(1)

    def run(self,dataset,keynum):
        if dataset.name[0:4]=='3dLo':
            datasetname = f'3d{dataset.name[4:]}'
        else:
            datasetname = dataset.name

        print(f'\nmatch the keypoints on {dataset.name}')
        Save_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        make_non_exists_dir(Save_dir)
        Feature_dir = f'{self.cfg.output_cache_fn}/{datasetname}/FCGF_feature'
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            feats0 = np.load(f'{Feature_dir}/{id0}.npy') #5000,32
            feats1 = np.load(f'{Feature_dir}/{id1}.npy') #5000,32
            feats0 = feats0/(np.sqrt(np.sum(np.square(feats0),axis=1,keepdims=True))+1e-5)
            feats1 = feats1/(np.sqrt(np.sum(np.square(feats1),axis=1,keepdims=True))+1e-5)

            sample0 = np.arange(feats0.shape[0])
            sample1 = np.arange(feats1.shape[0])
            np.random.shuffle(sample0)
            np.random.shuffle(sample1)
            sample0 = sample0[0:keynum]
            sample1 = sample1[0:keynum]

            feats0 = feats0[sample0]
            feats1 = feats1[sample1]
            feats0 = torch.from_numpy(np.transpose(feats0)[None,:,:]).cuda()
            feats1 = torch.from_numpy(np.transpose(feats1)[None,:,:]).cuda()
            d,argmin_of_0_in_1 = self.KNN(feats1,feats0)
            argmin_of_0_in_1 = argmin_of_0_in_1[0,0].cpu().numpy()
            d,argmin_of_1_in_0 = self.KNN(feats0,feats1)
            argmin_of_1_in_0 = argmin_of_1_in_0[0,0].cpu().numpy()
            match_pps = []
            for i in range(argmin_of_0_in_1.shape[0]):
                in0 = i
                in1 = argmin_of_0_in_1[i]
                inv_in0 = argmin_of_1_in_0[in1]
                if in0==inv_in0:
                    match_pps.append(np.array([[in0,in1]]))
            match_pps = np.concatenate(match_pps,axis=0) #n*2
            match_pps[:,0] = sample0[match_pps[:,0]]
            match_pps[:,1] = sample1[match_pps[:,1]]
            np.save(f'{Save_dir}/{id0}-{id1}.npy',match_pps)

def R_pre_log(dataset,save_dir):
    writer=open(f'{save_dir}/pre.log','w')
    pair_num=int(len(dataset.pc_ids))
    for pair in dataset.pair_ids:
        pc0,pc1=pair
        ransac_result=np.load(f'{save_dir}/{pc0}-{pc1}.npz',allow_pickle=True)
        transform_pr=ransac_result['trans']
        writer.write(f'{int(pc0)}\t{int(pc1)}\t{pair_num}\n')
        writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
        writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
        writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
        writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
    writer.close()

class estimator:
    def __init__(self,cfg):
        self.cfg = cfg
        self.inlier_dist = cfg.cal_trans_ird

    def Threepps2Trans(self,kps0_init,kps1_init):
        centre0 = np.mean(kps0_init,0,keepdims=True)
        centre1 = np.mean(kps1_init,0,keepdims=True)
        m = (kps1_init-centre1).T @ (kps0_init-centre0)
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T 
        offset =centre0 - (centre1 @ rotation.T)
        transform = np.concatenate([rotation,offset.T],1)
        transform = np.concatenate([transform,[[0.0,0.0,0.0,1.0]]],axis=0)
        return transform
    
    def ir_cal(self,key_m0,key_m1,T):
        key_m1 = transform_points(key_m1,T)
        diff = np.sum(np.square(key_m0-key_m1),axis=-1)
        ir_idx = [idx for idx,diff in enumerate(diff) if diff<self.inlier_dist*self.inlier_dist]
        ir_idx = np.array(ir_idx)
        # inliers0 = key_m0[ir_idx]
        # inliers1 = key_m1[ir_idx]
        inlier_ratio = np.mean(diff<self.inlier_dist*self.inlier_dist)
        return inlier_ratio,ir_idx
    
    def mse_cal(self,key_m0,key_m1,T):
        key_m1 = transform_points(key_m1,T)
        mse = np.mean(np.sum(np.square(key_m0-key_m1),axis=-1))
        return mse
    
    def mae_cal(self,key_m0,key_m1,T):
        key_m1 = transform_points(key_m1,T)
        mae = np.mean(np.sum(np.abs(key_m0-key_m1),axis=-1))
        return mae
    
    def chamfer_dist_cal(key_m0,key_m1,T):
        key_m1 = transform_points(key_m1,T)
        pc0 = o3d.geometry.PointCloud()
        pc1 = o3d.geometry.PointCloud()
        pc0.points = o3d.utility.Vector3dVector(key_m0)
        pc1.points = o3d.utility.Vector3dVector(key_m1)
        kdtree0 = o3d.geometry.KDTreeFlann(pc0)
        kdtree1 = o3d.geometry.KDTreeFlann(pc1)
        chamfer_distance_0to1 = 0.0
        chamfer_distance_1to0 = 0.0
        for point in key_m0:
            _, idx, _ = kdtree1.search_knn_vector_3d(point, 1)
            chamfer_distance_0to1 += np.linalg.norm(point - key_m1[idx[0]])
        for point in key_m1:
            _, idx, _ = kdtree0.search_knn_vector_3d(point, 1)
            chamfer_distance_1to0 += np.linalg.norm(point - key_m0[idx[0]])

        chamfer_distance = chamfer_distance_0to1 / len(key_m0) + chamfer_distance_1to0 / len(key_m1)
        return chamfer_distance


    def transdiff(self,gt,pre):
        Rdiff = compute_R_diff(gt[0:3:,0:3:],pre[0:3:,0:3:])
        tdiff = np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff
    
    def cal_trans(self,dataset,match_dir,Save_dir,pair):
        id0,id1 = pair

        Keys0 = dataset.get_kps(id0)
        Keys1 = dataset.get_kps(id1)

        pps = np.load(f'{match_dir}/{id0}-{id1}.npy')
        Keys_m0 = Keys0[pps[:,0]]
        Keys_m1 = Keys1[pps[:,1]]

        iter_cal = 0
        best_ir = 0
        best_ir_idx = []
        best_trans = np.eye(4)
        label = 0
        if self.cfg.score:
            top_trans = []
            # ir
            while iter_cal<self.cfg.max_iter:
                single_trans = {
                    'trans':[],
                    'inlier_ratio':float}
                iter_cal += 1
                idxs_init = np.random.choice(range(Keys_m0.shape[0]),3)
                kps0_init = Keys_m0[idxs_init]
                kps1_init = Keys_m1[idxs_init]

                trans = self.Threepps2Trans(kps0_init,kps1_init)
                inlier_ratio,ir_idx = self.ir_cal(Keys_m0,Keys_m1,trans)
                single_trans['trans'] = trans
                single_trans['inlier_ratio'] = inlier_ratio
                if iter_cal <= self.cfg.top_num:
                    top_trans.append(single_trans)
                else:
                    for i in range(self.cfg.top_num):
                        if single_trans['inlier_ratio'] > top_trans[i]['inlier_ratio']:
                            top_trans[i] = single_trans
                            break
            """ # for visual selection
            gt = dataset.get_transform(id0,id1)
            for i in range(len(top_trans)):
                trans = top_trans[i]['trans']
                Rdiff,tdiff = self.transdiff(gt,trans)
                top_trans[i]['rre'] = Rdiff
                top_trans[i]['rte'] = tdiff """
            top_trans = sorted(top_trans, key=lambda x:x["inlier_ratio"], reverse=True)
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',top_trans=top_trans)
            # MSE MAE Chamfer_dist
            """ while iter_cal<self.cfg.max_iter:
                single_trans = {
                    'trans':[],
                    'chd':float}
                iter_cal += 1
                idxs_init = np.random.choice(range(Keys_m0.shape[0]),3)
                kps0_init = Keys_m0[idxs_init]
                kps1_init = Keys_m1[idxs_init]

                trans = self.Threepps2Trans(kps0_init,kps1_init)
                # mse = self.mse_cal(Keys_m0,Keys_m1,trans)
                # mae = self.mae_cal(Keys_m0,Keys_m1,trans)
                chd = self.chamfer_dist_cal(Keys_m0,Keys_m1,trans)
                single_trans['trans'] = trans
                single_trans['chd'] = chd
                if iter_cal <= self.cfg.top_num:
                    top_trans.append(single_trans)
                else:
                    for i in range(self.cfg.top_num):
                        if single_trans['chd'] < top_trans[i]['chd']:
                            top_trans[i] = single_trans
                            break      
            top_trans = sorted(top_trans, key=lambda x:x["chd"])
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',top_trans=top_trans) """
        else:
            while iter_cal<self.cfg.max_iter:
                iter_cal += 1
                idxs_init = np.random.choice(range(Keys_m0.shape[0]),3)
                kps0_init = Keys_m0[idxs_init]
                kps1_init = Keys_m1[idxs_init]

                trans = self.Threepps2Trans(kps0_init,kps1_init)
                inlier_ratio,ir_idx = self.ir_cal(Keys_m0,Keys_m1,trans)
                if inlier_ratio>best_ir:
                    best_ir = inlier_ratio
                    best_ir_idx = ir_idx
                    best_trans = trans
            gt = dataset.get_transform(id0,id1)
            Rdiff,tdiff = self.transdiff(gt,best_trans)
            if Rdiff<self.cfg.label_R_th and tdiff<self.cfg.label_t_th:
                label = 1
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans = best_trans, ir = best_ir, label = label)     
            """ ir0 = Keys_m0[best_ir_idx]
            ir_ply0 = make_open3d_point_cloud(ir0)
            ir1 = Keys_m1[best_ir_idx]
            ir_ply1 = make_open3d_point_cloud(ir1)
            o3d.io.write_point_cloud(f'{Save_dir}/{id0}-{id1}-ir0.ply',ir_ply0)
            o3d.io.write_point_cloud(f'{Save_dir}/{id0}-{id1}-ir1.ply',ir_ply1) """

     
    def RANSAC(self,dataset):
        match_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/match_{self.cfg.keynum}'
        if self.cfg.score:
            Save_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/ir_top_trans'
            # Save_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/chd_top_trans'
        else:
            Save_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/trans'
        make_non_exists_dir(Save_dir)
        pair_ids = dataset.pair_ids

        print(f'\nCalculate the transformation of {dataset.name}')
        pool = Pool(len(pair_ids))
        func = partial(self.cal_trans,dataset,match_dir,Save_dir)
        list(tqdm(pool.imap(func,pair_ids),total=len(pair_ids)))
        pool.close()
        pool.join()
        """ for pair in tqdm(pair_ids):
            id0,id1=pair
            if os.path.exists(f'{Save_dir}/{id0}-{id1}.npz'):continue
            self.cal_trans(dataset,match_dir,Save_dir,pair) """
        print(f'\nCalculating top {self.cfg.top_num} transformations in {self.cfg.max_iter} iters has done.')

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

class score:
    def __init__(self,cfg):
        self.cfg = cfg
        self.point_limit = 30000
        self.neighbor_limits = np.array([41, 36, 34, 15])
        self.voxel_size = cfg.score_voxel
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
        input_dict = precompute_data_stack_mode(points, lengths, num_stages=4, voxel_size=self.voxel_size, 
                                                radius=0.0625, neighbor_limits = self.neighbor_limits, point_num=128)
        collated_dict.update(input_dict)
        return(collated_dict)

    def score(self, dataset):
        Save_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/ir_top_trans'
        trans_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/score_trans'
        # for feature vector check
        # vector_dir = f'{self.cfg.output_cache_fn}/{dataset.name}/feat_vectors'
        # make_non_exists_dir(vector_dir)
        make_non_exists_dir(trans_dir)
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            if os.path.exists(f'{trans_dir}/{id0}-{id1}.npz'):continue
            top_trans = np.load(f'{Save_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
            pcd0 = dataset.get_pc_o3d(id0)
            pcd0 = pcd0.voxel_down_sample(self.voxel_size)
            pcd0 = np.array(pcd0.points)
            if pcd0.shape[0] > self.point_limit:
                indices = np.random.permutation(pcd0.shape[0])[: self.point_limit]
                pcd0 = pcd0[indices]
            pcd1 = dataset.get_pc_o3d(id1)
            pcd1 = pcd1.voxel_down_sample(self.voxel_size)
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

            score = 0
            iter_time = 0
            trans_idx = 0
            save_trans = np.eye(4)
            save_score = 0
            save_overlap = 0
            while iter_time < self.cfg.max_time:
                if trans_idx >= len(top_trans):break
                trans = top_trans[trans_idx]['trans']
                overlap = top_trans[trans_idx]['inlier_ratio']
                trans_g = to_cuda(torch.from_numpy(trans))                
                cls_logits = self.score_model(collated_dict,ref_feats_c_norm,src_feats_c_norm,trans_g)
                # for feature vector check
                # dist_v = dist_v.detach().cpu().numpy()
                # feat_v = feat_v.detach().cpu().numpy()
                # np.save(f'{vector_dir}/{id0}-{id1}-{trans_idx}_dist.npy',dist_v)
                # np.save(f'{vector_dir}/{id0}-{id1}-{trans_idx}_feat.npy',feat_v)
                score = torch.sigmoid(cls_logits).detach().cpu().item()
                top_trans[trans_idx]['score'] = score
                torch.cuda.empty_cache()
                if score > save_score:
                    save_trans = trans
                    save_score = score
                    save_overlap = overlap
                iter_time += 1
                trans_idx += 1
            np.savez(f'{Save_dir}/{id0}-{id1}.npz', top_trans=top_trans)
            np.savez(f'{trans_dir}/{id0}-{id1}.npz', trans=save_trans, score=save_score, 
                     overlap=save_overlap, iter_time=iter_time)


class evaluator:
    def __init__(self,cfg):
        self.cfg = cfg
    def Feature_match_Recall(self,dataset,ratio=0.05):
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        pair_fmrs=[]
        irs = {}
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/match_{self.cfg.keynum}/{id0}-{id1}.npy')
            keys0=np.load(f'{Keys_dir}/cloud_bin_{id0}Keypoints.npy')[matches[:,0],:]
            keys1=np.load(f'{Keys_dir}/cloud_bin_{id1}Keypoints.npy')[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.cal_trans_ird) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)    
            irs[f'{id0}-{id1}'] = pair_fmr                          
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        FMR=np.mean(pair_fmrs>ratio)                                #FMR in one scene
        return FMR, pair_fmrs
    
class error_eval:
    def __init__(self, cfg):
        self.cfg = cfg
    def cal_error(self, datasets):
        recall,rre,rte = [],[],[]
        if self.cfg.ir:
            msg = f'{config.dataset}-ir\n'
        else:
            msg = f'{config.dataset}-score\n'
        for scene,dataset in tqdm(datasets.items()):
            if scene=='wholesetname':continue
            ok_num = 0
            scene_rre,scene_rte = [],[]
            for pair in tqdm(dataset.pair_ids):
                id0,id1=pair
                gt=dataset.get_transform(id0,id1)
                if self.cfg.ir:
                    top_trans = np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/ir_top_trans/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
                    pre = top_trans[0]['trans']
                else:
                    pre=np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/score_trans/{id0}-{id1}.npz')['trans']
                Rdiff = compute_R_diff(gt[0:3,0:3],pre[0:3,0:3])
                tdiff = np.linalg.norm(pre[0:3,-1]-gt[0:3,-1])
                if Rdiff<=self.cfg.label_R_th and tdiff<=self.cfg.label_t_th:
                    ok_num += 1
                    scene_rre.append(Rdiff)
                    scene_rte.append(tdiff)
            scene_recall = ok_num/len(dataset.pair_ids)
            scene_rre = np.mean(np.array(scene_rre))
            scene_rte = np.mean(np.array(scene_rte))
            msg += f'{scene}\trecall:{scene_recall:.4f}\tRRE:{scene_rre:.4f}\tRTE:{scene_rte:.4f}\n'
            recall.append(scene_recall)
            rre.append(scene_rre)
            rte.append(scene_rte)
        recall = np.mean(np.array(recall))
        rre = np.mean(np.array(rre))
        rte = np.mean(np.array(rte))
        msg += f'Recall:{recall:.4f}\nRRE:{rre:.4f}\nRTE:{rte:.4f}\n'
        with open(f'{self.cfg.output_cache_fn}/{self.cfg.dataset}/result.log','a') as f:
            f.write(msg)
        print(msg)



parser = argparse.ArgumentParser()
# registration parses
parser.add_argument('--dataset',default='3dmatch',type=str,help='dataset name')
parser.add_argument('--keynum',default=5000,type=int,help='number of key points')
parser.add_argument('--max_iter',default=5000,type=int,help='calculate transformation iterations')
parser.add_argument('--top_num',default=10,type=int,help='number of transformations contained')
parser.add_argument('--max_time',default=11,type=int)
parser.add_argument('--cal_trans_ird',default=0.1,type=float,help='inlier threshold of overlap calculation')
parser.add_argument('--label_R_th',default=15,type=float,help='rotation threshold for ture label')
parser.add_argument('--label_t_th',default=0.3,type=float,help='translation threshold for ture label')
parser.add_argument('--score',action='store_true')
parser.add_argument('--ir',action='store_true')
parser.add_argument('--weight',default='./Score_geo/weights/epoch-2.pth.tar',type=str)
parser.add_argument('--score_voxel',default=0.025,type=float)
# dir parses
base_dir='./data'
parser.add_argument('--origin_data_dir',type=str,default=f"{base_dir}/origin_data")
parser.add_argument('--output_cache_fn',type=str,default=f'{base_dir}/FCGF_Reg')
# eval parses
parser.add_argument('--fmr_ratio',default=0.05,type=float)
parser.add_argument('--RR_dist_th',default=0.2,type=float)
config = parser.parse_args()

matcer = matcher(config)
estmtor = estimator(config)
evaltor = evaluator(config)
error_evaltor = error_eval(config)
if config.score:
    scorer = score(config)

t1 = time.time()
datasets = get_dataset_name(config.dataset,config.origin_data_dir)
FMRS=[]
all_pair_fmrs=[]
make_non_exists_dir(f'{config.output_cache_fn}/{config.dataset}')
if config.score:
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        matcer.run(dataset)
        estmtor.RANSAC(dataset)
        print('Using Scorer-geo to score transformations.')
        scorer.score(dataset)
        R_pre_log(dataset,f'{config.output_cache_fn}/{dataset.name}/score_trans')
        print(f'eval the FMR result on {dataset.name}')
        FMR,pair_fmrs=evaltor.Feature_match_Recall(dataset,ratio=config.fmr_ratio)
        FMRS.append(FMR)
        all_pair_fmrs.append(pair_fmrs)
    FMRS=np.array(FMRS)
    all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)
    #RR
    if config.dataset == 'scannet':
        error_evaltor.cal_error(datasets)
    else:
        datasetname=datasets['wholesetname']
        Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(config,datasets,config.max_iter)
        t2 = time.time()
        t = t2-t1
        #print and save:
        msg=f'{datasetname}-score-{config.max_iter}iterations\n'
        msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
            f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
            f'Mean_Registration_Recall {Mean_Registration_Recall}\n' \
            f'time {t}\n'

        with open('data/results.log','a') as f:
            f.write(msg+'\n')
        print(msg)
else:
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        matcer.run(dataset)
        estmtor.RANSAC(dataset)
        R_pre_log(dataset,f'{config.output_cache_fn}/{dataset.name}/trans')
        print(f'eval the FMR result on {dataset.name}')
        FMR,pair_fmrs=evaltor.Feature_match_Recall(dataset,ratio=config.fmr_ratio)
        FMRS.append(FMR)
        all_pair_fmrs.append(pair_fmrs)
    FMRS=np.array(FMRS)
    all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)
    #RR
    datasetname=datasets['wholesetname']
    Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(config,datasets,config.max_iter)
    t2 = time.time()
    t = t2-t1
    #print and save:
    msg=f'{datasetname}-{config.max_iter}iterations\n'
    msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
        f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
        f'Mean_Registration_Recall {Mean_Registration_Recall}\n' \
        f'time {t}\n'

    with open('data/results.log','a') as f:
        f.write(msg+'\n')
    print(msg)
