
```
python regist.py --dataset 3dmatch --score
python regist.py --dataset 3dLomatch --score

python regist.py --dataset ETH --cal_trans_ird 0.2 --fmr_ratio 0.2 --RR_dist_th 0.5 --score_voxel 0.05 --score
python regist.py --dataset WHU-TLS --cal_trans_ird 1 --fmr_ratio 0.5 --RR_dist_th 1 --score_voxel 0.5 --score
python top_test.py --dataset WHU-TLS --RR_dist_th 1
```