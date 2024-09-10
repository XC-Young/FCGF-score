import torch
import torch.nn as nn
import torch.nn.functional as F

from Score_geo.utils.pairwise_distance import pairwise_distance
from Score_geo.transformer.geotransformer import (
    GeometricTransformer,
)

from Score_geo.backbone import KPConvFPN

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.dual_normalization = cfg.model.dual_normalization
        self.num_classes = cfg.model.num_classes

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GroupNorm(cfg.model.group_norm,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.GroupNorm(cfg.model.group_norm,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, data_dict):

        # Downsample point clouds
        feats = data_dict['features'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        points_c = data_dict['points'][-1].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]

        # 2. KPFCNN Encoder
        feats_c = self.backbone(feats, data_dict)

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        # 4. get the nearest point's spatial dist and feature dist
        dist_mat = torch.sqrt(pairwise_distance(ref_points_c,src_points_c))
        match_idx_ref = dist_mat.min(dim=1)[1] # index of each ref_point's NN point in src (Returns the index of min in each row)
        match_idx_src = dist_mat.min(dim=0)[1] # index of each src_point's NN point in ref (Returns the index of min in each column)
        min_dist_ref = dist_mat.min(dim=1)[0] # distance of ref_point to its NN point in src
        min_dist_src = dist_mat.min(dim=0)[0] # distance of src_point to its NN point in ref
        feat_match_score = torch.exp(-pairwise_distance(ref_feats_c_norm, src_feats_c_norm, normalized=True))
        feat_score_ref = feat_match_score[range(feat_match_score.shape[0]), match_idx_ref] # feature matching score of ref_point to its NN point in src
        feat_score_src = feat_match_score[match_idx_src, range(feat_match_score.shape[0])] # feature matching score of src_point to its NN point in ref

        # 5. classifier
        min_dist = torch.cat([min_dist_ref,min_dist_src])
        feat_score = torch.cat([feat_score_ref,feat_score_src])
        sorted_min_dist, perm = torch.sort(min_dist)
        sorted_feat_dist = feat_score[perm]
        # geo_out = torch.tensor([val for pair in zip(sorted_min_dist,sorted_feat_dist) for val in pair],device=torch.device("cuda")) # 512
        geo_out = torch.cat([sorted_min_dist,sorted_feat_dist])
        cls_logits = self.cls_head(geo_out.unsqueeze(0))
        return cls_logits


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg
    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
