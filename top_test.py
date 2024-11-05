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
def top_test_para():
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        Save_dir = f'{config.output_cache_fn}/{dataset.name}/ir_top_trans'
        trans_dir = f'{config.output_cache_fn}/{dataset.name}/ceiling_trans'
        make_non_exists_dir(trans_dir)
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            top_trans = np.load(f'{Save_dir}/{id0}-{id1}_trans.npy')
            top_irs = np.load(f'{Save_dir}/{id0}-{id1}_irs.npy')
            scores = np.load(f'{Save_dir}/{id0}-{id1}_scores.npy')
            gt = dataset.get_transform(id0,id1)
            # for ir_trans
            """ save_trans = top_trans[0]
            save_ir = top_irs[0]
            save_score = scores[0] """
            # for ceiling_trans
            for i in range(top_trans.shape[0]):
                trans = top_trans[i]
                Rdiff = compute_R_diff(gt[0:3:,0:3],trans[0:3:,0:3])
                tdiff = np.sqrt(np.sum(np.square(gt[0:3,3]-trans[0:3,3])))
                if i==0:
                    save_trans = trans
                    save_rre = Rdiff
                    save_ir = top_irs[i]
                    save_score = scores[i]
                elif i>0 and Rdiff<15 and tdiff<0.3:
                    if Rdiff<save_rre:
                        save_trans = trans
                        save_rre = Rdiff
                        save_ir = top_irs[i]
                        save_score = scores[i]
            np.savez(f'{trans_dir}/{id0}-{id1}.npz', trans=save_trans, inlier_ratio=save_ir, score=save_score)
        R_pre_log(dataset,f'{config.output_cache_fn}/{dataset.name}/ceiling_trans')
    Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(config,datasets,config.max_iter)
    datasetname=datasets['wholesetname']
    msg=f'{datasetname}-{config.max_iter}iterations\n'
    msg+=f'Mean_Registration_Recall {Mean_Registration_Recall}\n'
    with open('data/results.log','a') as f:
        f.write(msg+'\n')
    print(msg)

def top_test():
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        Save_dir = f'{config.output_cache_fn}/{dataset.name}/ir_top_trans'
        # trans_dir = f'{config.output_cache_fn}/{dataset.name}/ceiling_trans'
        trans_dir = f'{config.output_cache_fn}/{dataset.name}/ir_trans'
        make_non_exists_dir(trans_dir)
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            top_trans = np.load(f'{Save_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
            gt = dataset.get_transform(id0,id1)
            # for ir trans
            save_trans = top_trans[0]['trans']
            save_ir = top_trans[0]['inlier_ratio']
            save_score = top_trans[0]['score']
            # for top_trans ceiling
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
        R_pre_log(dataset,trans_dir)
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

class calerror:
    def __init__(self) -> None:
        pass

    def transdiff(self,gt,pre):
        Rdiff = compute_R_diff(gt[0:3:,0:3:],pre[0:3:,0:3:])
        tdiff = np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff
    
    def error_cal(self,dataset,top_trans_dir):
        scene_rre,scene_rte = [],[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1 = pair
            top_trans = np.load(f'{top_trans_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
            gt = dataset.get_transform(id0,id1)
            top_rre,top_rte = [],[]
            for i in range(len(top_trans)):
                trans = top_trans[i]['trans']
                Rdiff,tdiff = self.transdiff(gt,trans)
                top_rre.append(Rdiff)
                top_rte.append(tdiff)
            scene_rre.append(top_rre)
            scene_rte.append(top_rte)
        scene_rre = np.array(scene_rre) #n,top_num
        scene_rte = np.array(scene_rte)
        rre = np.mean(scene_rre,axis=0)
        rte = np.mean(scene_rte,axis=0)
        return rre,rte

def top_resort():
    errorcal = calerror()
    error_fn = f'{config.output_cache_fn}/{config.dataset}/{config.dataset}_error.log'
    writer = open(error_fn,'w')
    ir_rre,ir_rte = [],[]
    score_rre,score_rte = [],[]
    weight_rre,weight_rte = [],[]
    for scene,dataset in tqdm(datasets.items()):
        if scene=='wholesetname':continue
        if scene=='valscenes':continue
        Save_dir = f'{config.output_cache_fn}/{dataset.name}/ir_top_trans'
        score_trans_dir = f'{config.output_cache_fn}/{dataset.name}/score_top_trans'
        weight_trans_dir = f'{config.output_cache_fn}/{dataset.name}/weight_top_trans'
        make_non_exists_dir(score_trans_dir)
        make_non_exists_dir(weight_trans_dir)
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            top_trans = np.load(f'{Save_dir}/{id0}-{id1}.npz',allow_pickle=True)['top_trans']
            gt = dataset.get_transform(id0,id1)
            weight_top_trans = []
            for i in range(len(top_trans)):
                single = top_trans[i]
                ir = single['inlier_ratio']
                score = single['score']
                weight = score*ir
                single['weight'] = weight
                weight_top_trans.append(single)
            weight_top_trans = sorted(weight_top_trans, key=lambda x:x['weight'], reverse=True)
            score_sort_top_trans = sorted(top_trans, key=lambda x:x["score"], reverse=True)
            np.savez(f'{score_trans_dir}/{id0}-{id1}.npz',top_trans=score_sort_top_trans)
            np.savez(f'{weight_trans_dir}/{id0}-{id1}.npz',top_trans=weight_top_trans)

        ir_scene_rre,ir_scene_rte = errorcal.error_cal(dataset,Save_dir)
        score_scene_rre,score_scene_rte = errorcal.error_cal(dataset,score_trans_dir)
        weight_scene_rre,weight_scene_rte = errorcal.error_cal(dataset,score_trans_dir)
        ir_rre.append(ir_scene_rre)
        ir_rte.append(ir_scene_rte)
        score_rre.append(score_scene_rre)
        score_rte.append(score_scene_rte)
        weight_rre.append(weight_scene_rre)
        weight_rte.append(weight_scene_rte)
    ir_rre = np.mean(np.array(ir_rre),axis=0)
    ir_rte = np.mean(np.array(ir_rte),axis=0)
    score_rre = np.mean(np.array(score_rre),axis=0)
    score_rte = np.mean(np.array(score_rte),axis=0)
    weight_rre = np.mean(np.array(weight_rre),axis=0)
    weight_rte = np.mean(np.array(weight_rte),axis=0)
    writer.write(f'ir_rre\tir_rte\tscore_rre\tscore_rte\tweight_rre\tweight_rte\n')
    for i in range(ir_rre.shape[0]):
        writer.write(f'{ir_rre[i]}\t{ir_rte[i]}\t{score_rre[i]}\t{score_rte[i]}\t{weight_rre[i]}\t{weight_rte[i]}\n')
    writer.close() 

# top_test()
# score_false()
# top_resort()