import os,sys
import numpy as np
import random
import abc
import open3d as o3d
from utils.utils import make_non_exists_dir

class EvalDataset(abc.ABC):
    @abc.abstractmethod
    def get_pair_ids(self):
        pass

    @abc.abstractmethod
    def get_cloud_ids(self):
        pass

    @abc.abstractmethod
    def get_pc_dir(self,cloud_id):
        pass
    
    @abc.abstractmethod
    def get_key_dir(self,cloud_id):
        pass

    @abc.abstractmethod
    def get_transform(self,id0,id1):
        # note the order!
        # target: id0, source: id1
        # R @ pts1 + t = pts0
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_kps(self,cloud_id):
        pass

#The dataset class for original/ground truth datas
class ThrDMatchPartDataset(EvalDataset):
    def __init__(self,root_dir,stationnum,gt_dir=None):
        self.root=root_dir
        if gt_dir==None:
            self.gt_dir=f'{self.root}/PointCloud/gt.log'
        else:
            self.gt_dir=gt_dir
        self.kps_pc_fn=[f'{self.root}/Keypoints_PC/cloud_bin_{k}Keypoints.npy' for k in range(stationnum)]
        self.kps_fn=[f'{self.root}/Keypoints/cloud_bin_{k}Keypoints.txt' for k in range(stationnum)]
        self.pc_ply_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.ply' for k in range(stationnum)]
        self.pc_ply_vds_paths=[f'{self.root}/PointCloud_vds/cloud_bin_{k}.ply' for k in range(stationnum)]
        self.pc_txt_paths=[f'{self.root}/PointCloud/cloud_bin_{k}.txt' for k in range(stationnum)]
        self.pair_id2transform=self.parse_gt_fn(self.gt_dir)
        self.pair_ids=[tuple(v.split('-')) for v in self.pair_id2transform.keys()]
        self.pc_ids=[str(k) for k in range(stationnum)]
        self.pair_num=self.get_pair_nums()
        self.name='3dmatch/kitchen'

    #function for gt(input: gt.log)
    @staticmethod
    def parse_gt_fn(fn):
        with open(fn,'r') as f:
            lines=f.readlines()
            pair_num=len(lines)//5
            pair_id2transform={}
            for k in range(pair_num):
                id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep='\t')[0:2]
                id0=int(id0)
                id1=int(id1)
                row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
                row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
                row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
                transform=np.stack([row0,row1,row2],0)
                pair_id2transform['-'.join((str(id0),str(id1)))]=transform

            return pair_id2transform

    def get_pair_ids(self):
        return self.pair_ids

    def get_pair_nums(self):
        return len(self.pair_ids)

    def get_cloud_ids(self):
        return self.pc_ids

    def get_pc_dir(self,cloud_id):
        return self.pc_ply_paths[int(cloud_id)]

    def get_pc(self,pc_id):
        if os.path.exists(self.pc_ply_paths[int(pc_id)]):
            pc=o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
            return np.array(pc.points)
        else:
            pc=np.loadtxt(self.pc_paths[int(pc_id)],delimiter=',')
            return pc
    
    def get_pc_o3d(self,pc_id):
        return o3d.io.read_point_cloud(self.pc_ply_paths[int(pc_id)])
    
    def get_pc_vds_o3d(self,pc_id):
        return o3d.io.read_point_cloud(self.pc_ply_vds_paths[int(pc_id)])
            
    def get_key_dir(self,cloud_id):
        return self.kps_fn[int(cloud_id)]

    def get_transform(self, id0, id1):
        return self.pair_id2transform['-'.join((id0,id1))]

    def get_name(self):
        return self.name

    def get_kps(self, cloud_id):
        if not os.path.exists(self.kps_pc_fn[int(cloud_id)]):
            pc=self.get_pc(cloud_id)
            key_idxs=np.loadtxt(self.kps_fn[int(cloud_id)]).astype(np.int32)
            keys=pc[key_idxs]
            make_non_exists_dir(f'{self.root}/Keypoints_PC')
            np.save(self.kps_pc_fn[int(cloud_id)],keys)
            return keys
        return np.load(self.kps_pc_fn[int(cloud_id)])

#Get dataset items with the dataset name(output: dict)
def get_dataset_name(dataset_name,origin_data_dir):
    if dataset_name=='demo':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['kitchen']
        stationnums=[2]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='3dmatch':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=["kitchen","sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30","sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1","sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2","sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]
        stationnums=[60,60,60,55,57,37,66,38]

        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='3dLomatch':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=["kitchen","sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30","sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1","sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2","sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"]
        stationnums=[60,60,60,55,57,37,66,38]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/3dmatch/'+scenes[i]
            gt_dir=f'{root_dir}/PointCloud/gtLo.log'
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i],gt_dir)
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='ETH':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['gazebo_summer','gazebo_winter','wood_autumn','wood_summer']
        stationnums=[32,31,32,37]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets


    if dataset_name=='WHU-TLS':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['Park','Mountain','Campus','RiverBank','UndergroundExcavation','Tunnel']
        stationnums=[32,6,10,7,12,7]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='TLS_Dataset':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['SubwayStation','HighSpeedRailway','Mountain','Forest','Park','Campus','Residence','RiverBank','HeritageBuilding','UndergroundExcavation','Tunnel']
        stationnums=[6,8,6,5,32,10,7,7,9,12,7]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='scannet':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=["scene0197_01","scene0030_02","scene0406_02","scene0694_00",
                "scene0701_01","scene0457_01","scene0208_00","scene0578_01",
                "scene0286_02","scene0569_00","scene0309_00","scene0265_02",
                "scene0588_02","scene0474_01","scene0477_01","scene0334_02",
                "scene0353_00","scene0043_00","scene0224_00","scene0661_00",
                "scene0335_02","scene0231_01","scene0025_01","scene0642_02",
                "scene0493_01","scene0057_01","scene0575_02","scene0146_02",
                "scene0223_00","scene0262_01","scene0229_01","scene0676_01"]
        stationnums=[30]*len(scenes)
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/scannet/'+scenes[i]
            gt_dir=f'{root_dir}/PointCloud/gt_pair.log'
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i],gt_dir)
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='3dmatch_val':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        scenes=['sun3d-brown_bm_4-brown_bm_4','sun3d-harvard_c11-hv_c11_2','7-scenes-heads','rgbd-scenes-v2-scene_10','bundlefusion-office0','analysis-by-synthesis-apt2-kitchen']
        stationnums=[32,8,18,7,60,9]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    if dataset_name=='3dmatch_train':
        datasets={}
        datasets['wholesetname']=f'{dataset_name}'
        datasets['valscenes']=['sun3d-brown_bm_4-brown_bm_4','sun3d-harvard_c11-hv_c11_2','7-scenes-heads','rgbd-scenes-v2-scene_10','bundlefusion-office0','analysis-by-synthesis-apt2-kitchen']
        scenes=['bundlefusion-apt0', 'rgbd-scenes-v2-scene_02', 'bundlefusion-office1', 'sun3d-brown_cogsci_1-brown_cogsci_1', 'rgbd-scenes-v2-scene_06', 'analysis-by-synthesis-apt2-kitchen', 'rgbd-scenes-v2-scene_03', 'bundlefusion-apt1', 'sun3d-harvard_c8-hv_c8_3', 'bundlefusion-copyroom', 'sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika', 'rgbd-scenes-v2-scene_04', '7-scenes-pumpkin', 'rgbd-scenes-v2-scene_01', 'analysis-by-synthesis-office2-5a', 'sun3d-brown_bm_1-brown_bm_1', 'bundlefusion-apt2', 'sun3d-brown_cs_2-brown_cs2', 'bundlefusion-office2', 'sun3d-hotel_sf-scan1', 'sun3d-hotel_nips2012-nips_4', 'bundlefusion-office3', 'rgbd-scenes-v2-scene_09', 'rgbd-scenes-v2-scene_05', 'rgbd-scenes-v2-scene_07', '7-scenes-heads', 'sun3d-harvard_c3-hv_c3_1', 'rgbd-scenes-v2-scene_08', 'sun3d-mit_76_417-76-417b', 'sun3d-mit_32_d507-d507_2', 'sun3d-mit_46_ted_lab1-ted_lab_2', '7-scenes-chess', 'rgbd-scenes-v2-scene_10', 'sun3d-harvard_c11-hv_c11_2', 'analysis-by-synthesis-apt2-living', 'sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika', 'analysis-by-synthesis-apt1-living', 'analysis-by-synthesis-apt1-kitchen', 'sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika', '7-scenes-stairs', 'sun3d-brown_bm_4-brown_bm_4', 'bundlefusion-office0', 'sun3d-harvard_c6-hv_c6_1', 'rgbd-scenes-v2-scene_14', 'rgbd-scenes-v2-scene_12', 'analysis-by-synthesis-office2-5b', 'analysis-by-synthesis-apt2-luke', '7-scenes-office', 'sun3d-harvard_c5-hv_c5_1', 'sun3d-brown_cs_3-brown_cs3', '7-scenes-fire', 'rgbd-scenes-v2-scene_11', 'analysis-by-synthesis-apt2-bed', 'rgbd-scenes-v2-scene_13']
        stationnums=[85, 8, 57, 28, 10, 9, 8, 84, 10, 44, 96, 8, 54, 8, 14, 65, 38, 52, 34, 92, 62, 37, 7, 11, 9, 18, 19, 9, 77, 54, 75, 54, 7, 8, 10, 70, 15, 11, 26, 24, 32, 60, 15, 6, 7, 17, 19, 90, 20, 34, 36, 6, 10, 4]
        for i in range(len(scenes)):
            root_dir=f'{origin_data_dir}/{dataset_name}/'+scenes[i]
            datasets[scenes[i]]=ThrDMatchPartDataset(root_dir,stationnums[i])
            datasets[scenes[i]].name=f'{dataset_name}/{scenes[i]}'
        return datasets

    else:
        raise NotImplementedError
