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
        ceiling_trans=ransac_result['trans']
        writer.write(f'{int(pc0)}\t{int(pc1)}\t{pair_num}\n')
        writer.write(f'{ceiling_trans[0][0]}\t{ceiling_trans[0][1]}\t{ceiling_trans[0][2]}\t{ceiling_trans[0][3]}\n')
        writer.write(f'{ceiling_trans[1][0]}\t{ceiling_trans[1][1]}\t{ceiling_trans[1][2]}\t{ceiling_trans[1][3]}\n')
        writer.write(f'{ceiling_trans[2][0]}\t{ceiling_trans[2][1]}\t{ceiling_trans[2][2]}\t{ceiling_trans[2][3]}\n')
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
def top_test():
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        Save_dir = f'{config.output_cache_fn}/{dataset.name}/ir_top_trans'
        trans_dir = f'{config.output_cache_fn}/{dataset.name}/ceiling_trans'
        make_non_exists_dir(trans_dir)
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            top_trans = np.load(f'{Save_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
            gt = dataset.get_transform(id0,id1)
            # save_trans = top_trans[0]['trans']
            for i in range(len(top_trans)):
                trans = top_trans[i]['trans']
                Rdiff = compute_R_diff(gt[0:3:,0:3],trans[0:3:,0:3])
                tdiff = np.sqrt(np.sum(np.square(gt[0:3,3]-trans[0:3,3])))
                if i==0:
                    save_trans = trans
                    save_rre = Rdiff
                    save_ir = top_trans[i]['inlier_ratio']
                    save_score = top_trans[i]['score']
                elif i>0 and Rdiff<15 and tdiff<0.3:
                    if Rdiff<save_rre:
                        save_trans = trans
                        save_rre = Rdiff
                        save_ir = top_trans[i]['inlier_ratio']
                        save_score = top_trans[i]['score']
            np.savez(f'{trans_dir}/{id0}-{id1}.npz', trans=save_trans, inlier_ratio=save_ir, score=save_score)
        R_pre_log(dataset,f'{config.output_cache_fn}/{dataset.name}/ceiling_trans')
    Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(config,datasets,config.max_iter)
    datasetname=datasets['wholesetname']
    msg=f'{datasetname}-{config.max_iter}iterations\n'
    msg+=f'Mean_Registration_Recall {Mean_Registration_Recall}\n'
    with open('data/results.log','a') as f:
        f.write(msg+'\n')
    print(msg)

# find the ir_top right but score false
def score_false():
    num=0
    writer = open(f'{config.output_cache_fn}/{config.dataset}/score_false.log','w')
    for scene,dataset in datasets.items():
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        ceiling_dir = f'{config.output_cache_fn}/{dataset.name}/ceiling_trans'
        score_dir = f'{config.output_cache_fn}/{dataset.name}/weight_trans_2'
        for pair in dataset.pair_ids:
            id0,id1=pair
            ceiling = np.load(f'{ceiling_dir}/{id0}-{id1}.npz',allow_pickle=True)
            score = np.load(f'{score_dir}/{id0}-{id1}.npz',allow_pickle=True)
            gt = dataset.get_transform(id0,id1)
            ceiling_trans = ceiling['trans']
            score_trans = score['trans']
            c_rre = compute_R_diff(gt[0:3:,0:3],ceiling_trans[0:3:,0:3])
            s_rre = compute_R_diff(gt[0:3:,0:3],score_trans[0:3:,0:3])
            c_rte = np.sqrt(np.sum(np.square(gt[0:3,3]-ceiling_trans[0:3,3])))
            s_rte = np.sqrt(np.sum(np.square(gt[0:3,3]-score_trans[0:3,3])))
            if c_rre<15 and c_rte<0.3:
                if s_rre>=15 or s_rte>=0.3:
                    c_ir,c_score = ceiling['inlier_ratio'],ceiling['score']
                    s_ir,s_score = score['overlap'],score['score']
                    writer.write(f'{scene}-{id0}-{id1}\n')
                    writer.write(f'ceiling: rre {c_rre:.4f}, rte {c_rte:.4f}, ir {c_ir:.4f}, score {c_score:.4f}\n')
                    writer.write(f'{ceiling_trans[0][0]}\t{ceiling_trans[0][1]}\t{ceiling_trans[0][2]}\t{ceiling_trans[0][3]}\n')
                    writer.write(f'{ceiling_trans[1][0]}\t{ceiling_trans[1][1]}\t{ceiling_trans[1][2]}\t{ceiling_trans[1][3]}\n')
                    writer.write(f'{ceiling_trans[2][0]}\t{ceiling_trans[2][1]}\t{ceiling_trans[2][2]}\t{ceiling_trans[2][3]}\n')
                    writer.write(f'{ceiling_trans[3][0]}\t{ceiling_trans[3][1]}\t{ceiling_trans[3][2]}\t{ceiling_trans[3][3]}\n')
                    writer.write(f'weight : rre {s_rre:.4f}, rte {s_rte:.4f}, ir {s_ir:.4f}, score {s_score:.4f}; \n')
                    writer.write(f'{score_trans[0][0]}\t{score_trans[0][1]}\t{score_trans[0][2]}\t{score_trans[0][3]}\n')
                    writer.write(f'{score_trans[1][0]}\t{score_trans[1][1]}\t{score_trans[1][2]}\t{score_trans[1][3]}\n')
                    writer.write(f'{score_trans[2][0]}\t{score_trans[2][1]}\t{score_trans[2][2]}\t{score_trans[2][3]}\n')
                    writer.write(f'{score_trans[3][0]}\t{score_trans[3][1]}\t{score_trans[3][2]}\t{score_trans[3][3]}\n')
                    num += 1
    writer.close()
    print(num)

# top_test()
# score_false()