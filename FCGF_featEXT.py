import numpy as np
import torch
import argparse
from tqdm import tqdm
import open3d as o3d
import sys
sys.path.append('.')
from utils.utils import make_non_exists_dir,random_rotation_matrix,transform_points
from dataops.dataset import get_dataset_name
from backbone.model import load_model
from utils.knn_search import knn_module
from utils.utils_o3d import make_open3d_point_cloud
import MinkowskiEngine as ME

class data_pre():
    def __init__(self,config):
        self.cfg = config
        self.datasets = get_dataset_name(self.cfg.dataset,self.cfg.datadir)

 # add random rotation to 3dmatch_train
    def rot_pc(self):
        for scene,dataset in tqdm(self.datasets.items()):
            if scene in ['wholesetname','valscenes']:continue
            # rotate pc
            thdmt_Rot_dir = f'./data/origin_data/3dmatch_train_rot/{scene}/PointCloud/'
            make_non_exists_dir(thdmt_Rot_dir)
            rot_dir = f'{thdmt_Rot_dir}/rdm_R'
            make_non_exists_dir(rot_dir)
            for pc_id in tqdm(dataset.pc_ids):
                rdm_r = random_rotation_matrix()
                np.save(f'{rot_dir}/{pc_id}.npy',rdm_r)
                pc = dataset.get_pc(pc_id)
                pc_rot = transform_points(pc,rdm_r)
                ply = make_open3d_point_cloud(pc_rot)
                o3d.io.write_point_cloud(f'{thdmt_Rot_dir}/cloud_bin_{pc_id}.ply',ply)

            # generate rotated gt
            writer = open(f'{thdmt_Rot_dir}/gt.log','w')
            pair_num = int(len(dataset.pc_ids))
            for pair in tqdm(dataset.pair_ids):
                id0,id1 = pair
                gt = dataset.get_transform(id0,id1)
                # assert gt == np.identity(4)[0:3,:]
                r0 = np.load(f'{rot_dir}/{id0}.npy')
                r1 = np.load(f'{rot_dir}/{id1}.npy')
                r_gt = gt[0:3:,0:3:]
                R_rot = r0 @ r_gt @ r1.T
                gt[0:3:,0:3:] = R_rot
                gt_rot = gt
                writer.write(f'{int(id0)}\t{int(id1)}\t{pair_num}\n')
                writer.write(f'{gt_rot[0][0]}\t{gt_rot[0][1]}\t{gt_rot[0][2]}\t{gt_rot[0][3]}\n')
                writer.write(f'{gt_rot[1][0]}\t{gt_rot[1][1]}\t{gt_rot[1][2]}\t{gt_rot[1][3]}\n')
                writer.write(f'{gt_rot[2][0]}\t{gt_rot[2][1]}\t{gt_rot[2][2]}\t{gt_rot[2][3]}\n')
                writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
            writer.close()

    # generate keypoints with curature
    def pca_compute(self, data, sort=True):
        average_data = np.mean(data, axis=0)  # calculate the mean
        decentration_matrix = data - average_data  
        H = np.dot(decentration_matrix.T, decentration_matrix)  # solve for the covariance matrix H
        eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # use SVD to solve eigenvalues and eigenvectors

        if sort:
            sort = eigenvalues.argsort()[::-1]  # descending sort
            eigenvalues = eigenvalues[sort]  # index
        return eigenvalues
    def caculate_surface_curvature(self, cloud, radius=0.1):
        points = np.asarray(cloud.points)
        kdtree = o3d.geometry.KDTreeFlann(cloud)
        num_points = len(cloud.points)
        curvature = []  
        for i in range(num_points):
            k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)
            neighbors = points[idx, :]
            w = self.pca_compute(neighbors)  # w is the eigenvalue
            delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
            curvature.append(delt)
        curvature = np.array(curvature, dtype=np.float64)
        return curvature
    def curvature_kps(self):  
        # Downsampling keypoints based on curvature values
        kps_num = 5000
        for scene,dataset in tqdm(self.datasets.items()):
            if scene=='wholesetname':continue
            if scene=='valscenes':continue

            kps_idx_dir = f'{dataset.root}/Keypoints'
            kps_dir = f'{dataset.root}/Keypoints_PC'
            pc_vds = f'{dataset.root}/PointCloud_vds'
            make_non_exists_dir(kps_idx_dir)
            make_non_exists_dir(kps_dir)
            make_non_exists_dir(pc_vds)

            for pc_id in tqdm(dataset.pc_ids):
                pcd = dataset.get_pc_o3d(pc_id)
                pcd_down = pcd.voxel_down_sample(voxel_size = 0.03)
                pc_xyz = np.asarray(pcd_down.points)
                o3d.io.write_point_cloud(f'{pc_vds}/cloud_bin_{pc_id}.ply',pcd_down)
                surface_curvature = self.caculate_surface_curvature(pcd_down, radius=0.1)
                weight = surface_curvature/np.sum(surface_curvature)
                kps_idx = np.random.choice(pc_xyz.shape[0],kps_num,p=weight)
                kps = pc_xyz[kps_idx]
                np.savetxt(f'{kps_idx_dir}/cloud_bin_{pc_id}Keypoints.txt',kps_idx)
                np.save(f'{kps_dir}/cloud_bin_{pc_id}Keypoints.npy',kps)

class FCGFDataset():
    def __init__(self,datasets,config):
        self.cfg = config
        self.points={}
        self.pointlist=[]
        self.voxel_size = config.voxel_size
        self.datasets=datasets
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            for pc_id in dataset.pc_ids:
                self.pointlist.append((scene,pc_id))
                pts = self.datasets[scene].get_pc_o3d(pc_id)
                pts = pts.voxel_down_sample(config.voxel_size*0.4)
                pts = np.array(pts.points)
                self.points[f'{scene}_{pc_id}']=pts


    def __getitem__(self, idx):
        scene,pc_id=self.pointlist[idx]
        xyz0 = self.points[f'{scene}_{pc_id}']
        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        # Get coords
        xyz0 = np.array(pcd0.points)
        feats=np.ones((xyz0.shape[0], 1))
        coords0 = np.floor(xyz0 / self.voxel_size)
        
        return (xyz0, coords0, feats ,self.pointlist[idx])
    
    def __len__(self):
        return len(self.pointlist)
    
class FCGF_feat_ext():
    def __init__(self,config):
        self.config = config
        self.dataset_name = self.config.dataset
        self.output_dir = self.config.outdir
        self.origin_dir = self.config.datadir
        self.datasets = get_dataset_name(self.dataset_name,self.origin_dir)
        self.knn=knn_module.KNN(1)

    # extract batch feature of keypoints
    def collate_fn(self,list_data):
        xyz0, coords0, feats0, scenepc = list(
            zip(*list_data))
        xyz_batch0 = []
        dsxyz_batch0=[]
        batch_id = 0
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                raise ValueError(f'Can not convert to torch tensor, {x}')
        
        for batch_id, _ in enumerate(coords0):
            xyz_batch0.append(to_tensor(xyz0[batch_id]))
            _, inds = ME.utils.sparse_quantize(coords0[batch_id], return_index=True)
            dsxyz_batch0.append(to_tensor(xyz0[batch_id][inds]))

        coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)

        # Concatenate all lists
        xyz_batch0 = torch.cat(xyz_batch0, 0).float()
        dsxyz_batch0=torch.cat(dsxyz_batch0, 0).float()
        cuts_node=0
        cuts=[0]
        for batch_id, _ in enumerate(coords0):
            cuts_node+=coords0[batch_id].shape[0]
            cuts.append(cuts_node)

        return {
            'pcd0': xyz_batch0,
            'dspcd0':dsxyz_batch0,
            'scenepc':scenepc,
            'cuts':cuts,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
        }

    def Feature_extracting(self, data_loader):
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.config.model)
        config = checkpoint['config']
        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)    
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
         
        features={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            for pc_id in dataset.pc_ids:
                features[f'{scene}_{pc_id}']=[]

        with torch.no_grad():
            for i, input_dict in enumerate(tqdm(data_loader)):
                sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(device),
                        coordinates=input_dict['sinput0_C'].to(device))
                torch.cuda.synchronize()
                F0 = model(sinput0).F
                
                cuts=input_dict['cuts']
                scene_pc=input_dict['scenepc']
                for inb in range(len(scene_pc)):
                    scene,pc_id=scene_pc[inb]
                    make_non_exists_dir(f'{self.output_dir}/{self.dataset_name}/{scene}/FCGF_feature')
                    feature=F0[cuts[inb]:cuts[inb+1]]
                    pts=input_dict['dspcd0'][cuts[inb]:cuts[inb+1]]#*config.voxel_size

                    Keys_i=self.kps[f'{scene}_{pc_id}']
                    xyz_down=pts.T[None,:,:].cuda() #1,3,n
                    d,nnindex=self.knn(xyz_down,Keys_i)
                    nnindex=nnindex[0,0]
                    one_R_output=feature[nnindex,:].cpu().numpy()#5000*32
                                        
                    np.save(f'{self.output_dir}/{self.dataset_name}/{scene}/FCGF_feature/{pc_id}.npy',one_R_output)

    def batch_feature_extraction(self):
        #preload kps
        self.kps={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            if scene=='valscenes':continue
            for pc_id in dataset.pc_ids:
                kps = dataset.get_kps(pc_id)
                self.kps[f'{scene}_{pc_id}']=torch.from_numpy(kps.T[None,:,:].astype(np.float32)).cuda()
        dset=FCGFDataset(self.datasets,self.config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=4, # if out of memory change the batch_size to 1
            shuffle=False,
            num_workers=16,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False)
        self.Feature_extracting(loader)

if __name__ == '__main__':
    basedir = './data'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='./backbone/checkpoints/best_val_checkpoint.pth',
        type=str,
        help='path to latest backbone checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--dataset',
        default='3dmatch_train',
        help='datasetname')

    parser.add_argument(
        '--datadir',
        default=f'{basedir}/origin_data',
        type=str,
        help='dir for origindata')
    parser.add_argument(
        '--outdir',
        default=f'{basedir}/FCGF_Reg',
        type=str,
        help='path to output dir')
    
    parser.add_argument(
        '--pre_data',
        default=False,
        type=bool,
        help='Is pre-processing of data required'
    )
    
    args = parser.parse_args()
    if args.pre_data == True:
        data_prer = data_pre(args)
        data_prer.rot_pc()
        data_prer.curvature_kps()
    FCGF_feat_exter = FCGF_feat_ext(args)
    FCGF_feat_exter.batch_feature_extraction()

