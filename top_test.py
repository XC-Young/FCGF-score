import numpy as np
import argparse
from tqdm import tqdm
from utils.utils import make_non_exists_dir
from utils.r_eval import compute_R_diff
from dataops.dataset import get_dataset_name
import utils.RR_cal as RR_cal

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='3dLomatch',type=str,help='dataset name')
parser.add_argument('--max_iter',default=5000,type=int,help='calculate transformation iterations')
parser.add_argument('--score',action='store_true')
# dir parses
base_dir='./data'
parser.add_argument('--origin_data_dir',type=str,default=f"{base_dir}/origin_data")
parser.add_argument('--output_cache_fn',type=str,default=f'{base_dir}/FCGF_Reg')
parser.add_argument('--RR_dist_th',default=0.2)

config = parser.parse_args()
datasets = get_dataset_name(config.dataset,config.origin_data_dir)
for scene,dataset in tqdm(datasets.items()):
    if scene=='wholesetname':continue
    if scene=='valscenes':continue
    Save_dir = f'{config.output_cache_fn}/{dataset.name}/ir_top_trans'
    trans_dir = f'{config.output_cache_fn}/{dataset.name}/trans'
    make_non_exists_dir(trans_dir)
    for pair in tqdm(dataset.pair_ids):
        id0,id1=pair
        top_trans = np.load(f'{Save_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
        gt = dataset.get_transform(id0,id1)
        save_trans = np.eye(4)
        save_rre,save_rte = 500,500
        for i in range(len(top_trans)):
            trans = top_trans[i]['trans']
            Rdiff = compute_R_diff(gt[0:3:,0:3],trans[0:3:,0:3])
            tdiff = np.sqrt(np.sum(np.square(gt[0:3,3]-trans[0:3,3])))
            if i==0:
                save_trans = trans
                save_rre = Rdiff
                save_rte = tdiff
            elif i>0 and Rdiff<15 and tdiff<0.3:
                if Rdiff<save_rre:
                    save_trans = trans
                    save_rre = Rdiff
                    save_rte = tdiff
        np.savez(f'{trans_dir}/{id0}-{id1}.npz', trans=save_trans)
    R_pre_log(dataset,f'{config.output_cache_fn}/{dataset.name}/trans')
Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(config,datasets,config.max_iter)
datasetname=datasets['wholesetname']
msg=f'{datasetname}-{config.max_iter}iterations\n'
msg+=f'Mean_Registration_Recall {Mean_Registration_Recall}\n'
with open('data/results.log','a') as f:
    f.write(msg+'\n')
print(msg)
