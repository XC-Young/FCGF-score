import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
from dataops.dataset import get_dataset_name
import MinkowskiEngine as ME

def collate_batch_fn(list_data):
  feats,coords,label = list(
      zip(*list_data))
  label_batch= []

  batch_id = 0
  curr_start_inds = np.zeros((1, 1))

  def to_tensor(x):
    if isinstance(x, torch.Tensor):
      return x
    elif isinstance(x, np.ndarray):
      return torch.from_numpy(x)
    else:
      raise ValueError(f'Can not convert to torch tensor, {x}')

  for batch_id, _ in enumerate(coords):
    N = coords[batch_id].shape[0]
    label_batch.append(label[batch_id])
    # Move the head
    curr_start_inds[0] += N
  coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats) #coords_batch0:n*4; feats_batch0:n*2

  # Concatenate all lists
#   label_batch = torch.cat(label_batch, 0).int()

  return (feats_batch,coords_batch,label_batch)

# train dataset
class Train_dataset(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg = cfg 
        self.dataset_name = self.cfg.trainset_name
        self.origin_data_dir = f'{self.cfg.base_dir}/data/origin_data'
        self.trans_dir = self.cfg.train_trans_dir
        self.datasets = get_dataset_name(self.dataset_name,self.origin_data_dir)
        self.valscenes = self.datasets['valscenes']
        self.is_training = is_training
        self.pairs = [] #list: scene_name id0 id1 tran_idx
        if self.is_training:
            for scene,dataset in self.datasets.items():
                if scene in ['wholesetname','valscenes']:continue
                for pair in dataset.pair_ids:
                    id0,id1 = pair
                    for i in range(self.cfg.trans_num):
                        self.pairs.append([scene,id0,id1,i])
        else:
            for scene in self.valscenes:
                dataset = self.datasets[scene]
                for pair in dataset.pair_ids:
                    id0,id1 = pair
                    for i in range(self.cfg.trans_num):
                        self.pairs.append([scene,id0,id1,i])

    def get_matches(self,pcd0,pcd1,threshold=0.1):
        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        ir_idx0 = []
        for i,point in enumerate(pcd0.points):
            [k,idx,_] = pcd1_tree.search_radius_vector_3d(point,threshold)
            if(k!=0):
                ir_idx0.append(i)
        ir_idx0 = np.asarray(ir_idx0)
        return(ir_idx0)

    def __getitem__(self, index):
        scene,id0,id1,tran_idx = self.pairs[index]
        pcd0 = o3d.io.read_point_cloud(f'{self.origin_data_dir}/{self.dataset_name}/{scene}/PointCloud_vds/cloud_bin_{id0}.ply')
        pcd1 = o3d.io.read_point_cloud(f'{self.origin_data_dir}/{self.dataset_name}/{scene}/PointCloud_vds/cloud_bin_{id1}.ply')
        pc0 = np.array(pcd0.points)
        pc1 = np.array(pcd1.points)
        key_idx0 = np.loadtxt(f'{self.origin_data_dir}/{self.dataset_name}/{scene}/Keypoints/cloud_bin_{id0}Keypoints.txt').astype(np.int64)
        key_idx1 = np.loadtxt(f'{self.origin_data_dir}/{self.dataset_name}/{scene}/Keypoints/cloud_bin_{id1}Keypoints.txt').astype(np.int64)
        trans = np.load(f'{self.trans_dir}/{self.dataset_name}/{scene}/match_{self.cfg.key_num}/trans/{id0}-{id1}-trans.npy') #(trans_num,4,4)
        labels = np.load(f'{self.trans_dir}/{self.dataset_name}/{scene}/match_{self.cfg.key_num}/trans/{id0}-{id1}-label.npy') #(trans_num,)
        tran = trans[tran_idx]
        label = int(labels[tran_idx])
        # generate xyzkm
        xyzkm0 = np.c_[pc0,np.zeros(pc0.shape[0]).T,np.zeros(pc0.shape[0]).T]
        xyzkm1 = np.c_[pc1,np.zeros(pc1.shape[0]).T,np.zeros(pc1.shape[0]).T]
        pcd1 = pcd1.transform(tran)
        match_idxs0 = self.get_matches(pcd0,pcd1,self.cfg.match_th) #相当于得到是重叠区域？
        match_idxs1 = self.get_matches(pcd1,pcd0,self.cfg.match_th)
        for m in range(pc0.shape[0]):
            if m in key_idx0:
                xyzkm0[m,3] = 1
            if m in match_idxs0:
                xyzkm0[m,4] = 1
        for n in range(pc1.shape[0]):
            if n in key_idx1:
                xyzkm1[n,3] = 1
            if n in match_idxs1:
                xyzkm1[n,4] = 1
        xyzkm = np.concatenate((xyzkm0,xyzkm1),axis=0) #(n+m)*5
        # to SparseTensor
        feats = xyzkm[:,3:5].astype(np.float32)
        # voxelization
        coords = np.floor(xyzkm[:,0:3] / self.cfg.voxel_size).astype(np.int32)
        # remove the duplicates
        coords, idxs = ME.utils.sparse_quantize(coords, return_index=True)
        feats = feats[idxs]
        return(feats,coords,label)
        
    def __len__(self):
        return len(self.pairs)