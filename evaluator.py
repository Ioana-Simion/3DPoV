import copy
import torch
import numpy as np
from torchmetrics import Metric
from typing import Optional, List, Tuple, Dict
import os
import time
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from data_loader import PascalVOCDataModule
import wandb
from timm.models import create_model
from models import FCNHead
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from eval_metrics import PredsmIoU
from clustering import PerDatasetClustering
import torch.nn.functional as nn_F
from tqdm import tqdm
from einops import einsum
from torch.nn.functional import cosine_similarity

import faiss
import faiss.contrib.torch_utils

class LinearFinetune(torch.nn.Module):
    def __init__(self, model, num_classes: int, lr: float, input_size: int, spatial_res: int, val_iters: int,
                 drop_at: int, arch: str, head_type: str = None, decay_rate: float = 0.1, ignore_index: int = 255, device=None):
        super().__init__()
        # Init Model
        # if 'vit' in arch:
            # self.model = create_model(f'{arch}_patch{patch_size}_224', pretrained=False)
        self.model = model
        self.model_embed_dim = self.model.d_model
        if head_type == "fcn":
            self.finetune_head = FCNHead(
                in_channels=self.model_embed_dim,
                channels=512,
                num_convs=2,
                concat_input=True,
                dropout_ratio=0.1,
                num_classes=num_classes,
            )
        else:
            self.finetune_head = torch.nn.Conv2d(self.model_embed_dim, num_classes, 1)
        
        self.finetune_head = self.finetune_head.to(device)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.val_iters = val_iters
        self.input_size = input_size
        self.spatial_res = spatial_res
        self.drop_at = drop_at
        self.arch = arch
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.train_mask_size = 100
        self.val_mask_size = 100
        self.device = device
        self.optimizer, self.scheduler = self.configure_optimizers()

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.finetune_head.parameters(), weight_decay=0.0001,
                                    momentum=0.9, lr=self.lr)
        scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        return optimizer, scheduler

    def train_step(self, batch):
        self.finetune_head.train()
        imgs, masks = batch
        imgs = imgs.to(self.device)
        masks = masks.to(self.device)
        bs = imgs.size(0)
        res = imgs.size(3)
        assert res == self.input_size
        self.model.eval()

        with torch.no_grad():
            tokens, _ = self.model.forward_features(imgs)
            if 'vit' in self.arch:
                tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.model_embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.train_mask_size, self.train_mask_size),
                                               mode='bilinear')
        mask_preds = self.finetune_head(tokens)

        masks *= 255
        if self.train_mask_size != self.input_size:
            with torch.no_grad():
                masks = nn.functional.interpolate(masks, size=(self.train_mask_size, self.train_mask_size),
                                                  mode='nearest')

        loss = self.criterion(mask_preds, masks.long().squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def validation_step(self, batch):
        self.finetune_head.eval()
        with torch.no_grad():
            imgs, masks = batch
            bs = imgs.size(0)
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            tokens, _ = self.model.forward_features(imgs)
            tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.model_embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.val_mask_size, self.val_mask_size),
                                                mode='bilinear')
            mask_preds = self.finetune_head(tokens)

            # downsample masks and preds
            gt = masks * 255
            gt = nn.functional.interpolate(gt, size=(self.val_mask_size, self.val_mask_size), mode='nearest')
            valid = (gt != self.ignore_index) # mask to remove object boundary class
            mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

            # update metric
            self.miou_metric.update(gt[valid], mask_preds[valid])

    def validation_epoch_end(self):
        miou = self.miou_metric.compute(True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        return miou
    



class LinearFinetuneModule():
    def __init__(self, model, train_dataloader, val_dataloader, device, spatial_resolution=14, train_epoch=20, drop_at=20):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.train_epoch = train_epoch
        self.spatial_resolution = spatial_resolution
        self.drop_at = drop_at
        total_iters = len(self.train_dataloader) * self.drop_at
        cloned_model = copy.deepcopy(self.model)
        self.linear_evaluator = LinearFinetune(cloned_model,  num_classes=21, lr=0.01, input_size=224, spatial_res=self.spatial_resolution, val_iters=20,
                    drop_at=total_iters, arch="vit_small", head_type="lc", device=self.device)

    def linear_segmentation_validation(self):

        ## keep a dictionary of a few parameters of the model and later check if they are changed
        ########################################################
        # dict = {}
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         dict[name] = param.data.clone()

        ########################################################
        
        ########################################################
        ## check if the parameters are changed
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         assert torch.equal(param.data, dict[name])

        ########################################################
        

        print("==============================================================")
        final_miou = 0
        for j in range(self.train_epoch):
            for i, (x, y) in enumerate(self.train_dataloader):
                loss = self.linear_evaluator.train_step((x, y))
                print('linear_eval_loss', loss)
        for i, (x, y) in enumerate(self.val_dataloader):
            self.linear_evaluator.validation_step((x, y))
        miou = self.linear_evaluator.validation_epoch_end()
        final_miou = miou
        print('miou_val', round(miou, 6))
        return final_miou



def argmax_2d(x, max_value=True):
    h, w = x.shape[-2:]
    x = torch.flatten(x, start_dim=-2)
    if max_value:
        flat_indices = x.argmax(dim=-1)
    else:
        flat_indices = x.argmin(dim=-1)

    min_row = flat_indices // w
    min_col = flat_indices % w
    xy_indices = torch.stack((min_col, min_row), dim=-1)
    return xy_indices


class KeypointMatchingModule():
    def __init__(self, model, dataset, device, threshold=0.10):
        """
        Initialize the Keypoint Matching Evaluation module.
        Args:
            model: The feature extractor model to evaluate.
            dataset: The SPair dataset for keypoint matching evaluation.
            device: Device to perform computations on ('cuda' or 'cpu').
            threshold: Distance threshold for keypoint matching success.
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_errors(self, instance, mask_feats=False, return_heatmaps=False):
        img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, _ = instance
        mask_i = torch.tensor(np.array(mask_i, dtype=float))
        mask_j = torch.tensor(np.array(mask_j, dtype=float))

        images = torch.stack((img_i, img_j)).cuda()
        masks = torch.stack((mask_i, mask_j)).cuda()
        masks = torch.nn.functional.avg_pool2d(masks.float(), 16)
        masks = masks > 4 / (16**2)

        #feats = model(images)
        feats, _ = self.model.feature_extractor.forward_features(images)
        
        # DO NOT DO THIS
        # if self.model.model_type == "registers":
        #         # Exclude registers during validation
        #         feats = feats[:, :-8, :]  # Last 8 are registers
        if isinstance(feats, dict):
            print("using patchtokens")
            feats = feats.get("x_norm_patchtokens", feats)
        # if isinstance(feats, list):
        #     print("using list")
        #     feats = torch.cat(feats, dim=1)
        batch_size, num_patches, channels = feats.shape
        #print(f"batch_size {batch_size} num_patches {num_patches}, channels {channels}")
        grid_size = int(num_patches ** 0.5)


        if grid_size ** 2 == num_patches:
            # Directly reshape since it forms a square grid
            feats = feats.view(batch_size, grid_size, grid_size, channels).permute(0, 3, 1, 2)
            #print(f"Final reshaped feature shape (no CLS token): {feats.shape}")

        # Case 2: CLS token is included, remove it and reshape
        elif grid_size ** 2 == num_patches - 1:
            #print("Removing the CLS token to align shapes.")
            feats = feats[:, 1:]  # Remove CLS token
            batch_size, num_patches, channels = feats.shape
            grid_size = int(num_patches ** 0.5)
            feats = feats.view(batch_size, grid_size, grid_size, channels).permute(0, 3, 1, 2)

        feats = nn_F.normalize(feats, p=2, dim=1)


        if mask_feats:
            feats = feats * masks

        feats_i = feats[0]
        feats_j = feats[1]

        # normalize kps to [0, 1]
        assert images.shape[-1] == images.shape[-2], "assuming square images here"
        kps_i = kps_i.float()
        kps_j = kps_j.float()
        kps_i[:, :2] = kps_i[:, :2] / images.shape[-1]
        kps_j[:, :2] = kps_j[:, :2] / images.shape[-1]

        # get correspondences
        kps_i_ndc = (kps_i[:, :2].float() * 2 - 1)[None, None].cuda()
        kp_i_F = nn_F.grid_sample(
            feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
        )
        kp_i_F = kp_i_F[0, :, 0].t()

        # get max index in [0,1] range
        heatmaps = einsum(kp_i_F, feats_j, "k f, f h w -> k h w")
        pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

        # compute error and scale to threshold (for all pairs)
        errors = (pred_kp[:, None, :] - kps_j[None, :, :2]).norm(p=2, dim=-1)
        errors = errors / thresh_scale

        # only retain keypoints in both (for now)
        valid_kps = (kps_i[:, None, 2] * kps_j[None, :, 2]) == 1
        in_both = valid_kps.diagonal()

        # max error should be 1, so this excludes invalid from NN-search
        errors[valid_kps.logical_not()] = 1e3

        error_same = errors.diagonal()[in_both]
        error_nn, index_nn = errors[in_both].min(dim=1)
        index_same = in_both.nonzero().squeeze(1)

        if return_heatmaps:
            return error_same, error_nn, index_same, index_nn, heatmaps
        else:
            return error_same, error_nn, index_same, index_nn

    @torch.no_grad()
    def evaluate_dataset(self, dataset, thresh, verbose=False):
        pbar = tqdm(range(len(dataset)), ncols=60) if verbose else range(len(dataset))
        error_output = [self.compute_errors(self.dataset[i]) for i in pbar]

        errors = torch.cat([_err[0] for _err in error_output])
        src_ind = torch.cat([_err[2] for _err in error_output])
        tgt_ind = torch.cat([_err[3] for _err in error_output])

        # compute confusion matrix
        kp_max = max(src_ind.max(), tgt_ind.max()) + 1
        confusion = torch.zeros((kp_max, kp_max))
        for src, tgt in torch.stack((src_ind, tgt_ind), dim=1):
            confusion[src, tgt] += 1

        # compute recall
        recall = (errors < thresh).float().mean().item() * 100.0

        return recall, confusion
    
    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on keypoint matching over the dataset.
        Returns:
            recall: Keypoint matching recall score.
        """
        all_errors = []
        for i in tqdm(range(len(self.dataset)), ncols=60):
            errors_same, errors_nn = self.compute_errors(self.dataset[i])
            all_errors.extend(errors_same)
        
        recall = (torch.tensor(all_errors) < self.threshold).float().mean().item() * 100.0
        return recall


res = faiss.StandardGpuResources() # reuse this
res.setTempMemory(128 * 1024 * 1024) # 128MB

def faiss_knn(query, target, k):
    # make sure query and target are contiguous
    query = query.contiguous()
    target = target.contiguous()

    num_elements, feat_dim = query.shape
    gpu_index = faiss.GpuIndexFlatL2(res, feat_dim)
    gpu_index.add(target)
    dist, index = gpu_index.search(query, k)
    return dist, index


def knn_points(X_f, Y_f, K=1, metric="euclidean"):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.
    """
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine":
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    _, X_nn = faiss_knn(X_f, Y_f, K)

    # n_points x k x F
    X_f_nn = Y_f[X_nn]

    if metric == "euclidean":
        dists = (X_f_nn - X_f[:, None, :]).norm(p=2, dim=3)
    elif metric == "cosine":
        dists = 1 - cosine_similarity(X_f_nn, X_f[:, None, :], dim=-1)

    return dists, X_nn


def get_correspondences_ratio_test(
    P1_F, P2_F, num_corres, metric="cosine", bidirectional=False, ratio_test=True
):
    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    # output is cosine distance (0, 2)
    K = 2

    dists_1, idx_1 = knn_points(P1_F, P2_F, K, metric)
    idx_1 = idx_1[..., 0]
    if ratio_test:
        weights_1 = calculate_ratio_test(dists_1)
    else:
        weights_1 = dists_1[:, 0]

    # Take the nearest neighbor for the indices for k={1, 2}
    if bidirectional:
        dists_2, idx_2 = knn_points(P2_F, P1_F, K, metric)
        idx_2 = idx_2[..., 0]
        if ratio_test:
            weights_2 = calculate_ratio_test(dists_2)
        else:
            weights_2 = dists_2[:, 0]

        # Get topK matches in both directions
        m12_idx1, m12_idx2, m12_dist = get_topk_matches(
            weights_1, idx_1, num_corres // 2
        )
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(
            weights_2, idx_2, num_corres // 2
        )

        # concatenate into correspondences and weights
        all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)
        all_dist = torch.cat((m12_dist, m21_dist), dim=1)
    else:
        all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, num_corres)

    return all_idx1, all_idx2, all_dist


@torch.jit.script
def calculate_ratio_test(dists: torch.Tensor):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    # clamping because some dists will be 0 (when not in the pointcloud
    dists = dists.clamp(min=1e-9)
    ratio = dists[..., 0] / dists[..., 1].clamp(min=1e-9)
    weight = 1 - ratio
    return weight


# @torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    num_corres = min(num_corres, dists.shape[-1])
    dist, idx_source = torch.topk(dists, k=num_corres, dim=-1)
    idx_target = idx[idx_source]
    return idx_source, idx_target, dist


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


def grid_to_pointcloud(K_inv, depth, grid=None):
    _, H, W = depth.shape

    if grid is None:
        grid = get_grid(H, W)

    # Apply inverse projection
    points = depth * grid

    # Invert intriniscs
    points = points.view(3, H * W)
    points = K_inv @ points
    points = points.permute(1, 0)

    return points


def sample_pointcloud_features(feats, K, pc, image_shape):
    H, W = image_shape
    uvd = pc @ K.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)

    uv[:, 0] = (2 * uv[:, 0] / W) - 1
    uv[:, 1] = (2 * uv[:, 1] / H) - 1

    #TODO verify
    num_patches, dim = feats.shape
    grid_size = int(num_patches**0.5)
    assert grid_size * grid_size == num_patches, "Non-square grid"

    feats = feats.view(grid_size, grid_size, dim).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    grid = uv.view(1, -1, 1, 2)  # [1, N, 1, 2]

    pc_F = nn_F.grid_sample(feats, grid, mode="bilinear", align_corners=False)
    pc_F = pc_F.squeeze(0).squeeze(2).transpose(0, 1)  # [N, C]

    # # sample points
    # pc_F = nn_F.grid_sample(feats[None], uv[None, None], align_corners=False)
    # pc_F = pc_F[:, :, 0].transpose(1, 2)[0]

    return pc_F


def argmax_2d(x, max_value=True):
    h, w = x.shape[-2:]
    x = torch.flatten(x, start_dim=-2)
    if max_value:
        flat_indices = x.argmax(dim=-1)
    else:
        flat_indices = x.argmin(dim=-1)

    min_row = flat_indices // w
    min_col = flat_indices % w
    xy_indices = torch.stack((min_col, min_row), dim=-1)
    return xy_indices


def project_3dto2d(xyz, K_mat):
    uvd = xyz @ K_mat.transpose(-1, -2)
    uv = uvd[:, :2] / uvd[:, 2:3].clamp(min=1e-9)
    return uv


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return aucs


def estimate_correspondence_depth(feat_0, feat_1, depth_0, depth_1, K, num_corr=500):
    xyz_0 = grid_to_pointcloud(K.inverse(), depth_0)
    xyz_1 = grid_to_pointcloud(K.inverse(), depth_1)
    xyz_0 = xyz_0[xyz_0[:, 2] > 0]
    xyz_1 = xyz_1[xyz_1[:, 2] > 0]

    #print(f'depth_0 {depth_0.shape} depth_1 {depth_1.shape} xyz_0 {xyz_0.shape} xyz_1 {xyz_1.shape}')
    feat_0 = sample_pointcloud_features(feat_0, K.clone(), xyz_0, depth_0.shape[-2:])
    feat_1 = sample_pointcloud_features(feat_1, K.clone(), xyz_1, depth_1.shape[-2:])

    idx0, idx1, corr_dist = get_correspondences_ratio_test(feat_0, feat_1, num_corr)

    corr_xyz0 = xyz_0[idx0]
    corr_xyz1 = xyz_1[idx1]

    return corr_xyz0, corr_xyz1, corr_dist


def estimate_correspondence_xyz(
    feat_0, feat_1, xyz_grid_0, xyz_grid_1, num_corr=500, ratio_test=True
):
    # upsample feats
    _, h, w = xyz_grid_0.shape
    feat_0 = nn_F.interpolate(feat_0[None], size=(h, w), mode="bicubic")[0]
    feat_1 = nn_F.interpolate(feat_1[None], size=(h, w), mode="bicubic")[0]

    uvd_0 = get_grid(h, w).to(xyz_grid_0)
    uvd_1 = get_grid(h, w).to(xyz_grid_1)

    # only keep values with real points
    feat_0 = feat_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    feat_1 = feat_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    xyz_0 = xyz_grid_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    xyz_1 = xyz_grid_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]
    uvd_0 = uvd_0.permute(1, 2, 0)[xyz_grid_0[2] > 0]
    uvd_1 = uvd_1.permute(1, 2, 0)[xyz_grid_1[2] > 0]

    idx0, idx1, c_dist = get_correspondences_ratio_test(
        feat_0, feat_1, num_corr, ratio_test=ratio_test
    )

    c_xyz0 = xyz_0[idx0]
    c_xyz1 = xyz_1[idx1]
    c_uv0 = uvd_0[idx0][:, :2]
    c_uv1 = uvd_1[idx1][:, :2]

    return c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1


def compute_binned_performance(y, x, x_bins):
    """
    Given two arrays: (x, y), compute the mean y value for specific x_bins
    """
    y_binned = []
    for i in range(len(x_bins) - 1):
        x_min = x_bins[i]
        x_max = x_bins[i + 1]
        x_mask = (x >= x_min) * (x < x_max)
        y_binned.append(y[x_mask].mean())

    return y_binned

def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    R = viewpoint[..., :3, :3]
    t = viewpoint[..., None, :3, 3]
    # N.B. points is (..., n, 3) not (..., 3, n)
    if inverse:
        return (points - t) @ R
    else:
        return points @ R.transpose(-2, -1) + t

class ScanNetCorrespondenceModule():
    def __init__(self, model, dataset, device, num_corr=1000, scale_factor=0.25):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_corr = num_corr
        self.scale_factor = scale_factor
        self.model.to(self.device).eval()

    def evaluate(self):
        self.model.eval()
        err_2d = []
        
        for instance in tqdm(self.dataset, desc="ScanNet Eval"):
            rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"])).cuda()
            depths = torch.stack((instance["depth_0"], instance["depth_1"])).cuda()
            K = instance["K"].clone()
            Rt_gt = instance["Rt_1"].float()[:3, :4].cpu()

            rgbs = nn_F.interpolate(rgbs, size=(224, 224), mode="bilinear", align_corners=False)
            depths = nn_F.interpolate(depths, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            K[:2] *= self.scale_factor

            feats, _ = self.model.feature_extractor.forward_features(rgbs)
            feats = feats["x_norm_patchtokens"] if isinstance(feats, dict) else feats
            if feats.shape[1] == 1 + (224 // 14)**2:
                feats = feats[:, 1:]
            feats = feats.detach().cpu()

            xyz0, xyz1, corr_dist = estimate_correspondence_depth(
                feats[0], feats[1], depths[0].cpu(), depths[1].cpu(), K.clone().cpu(), self.num_corr
            )

            xyz0_in1 = transform_points_Rt(xyz0, Rt_gt)
            uv0 = project_3dto2d(xyz0_in1, K.cpu())
            uv1 = project_3dto2d(xyz1, K.cpu())

            corr_err2d = (uv0 - uv1).norm(p=2, dim=1)
            err_2d.append(corr_err2d.cpu())

        err_2d = torch.cat(err_2d, dim=0).float()
        recall_10px = (err_2d < 10).float().mean().item() * 100.0
        return recall_10px
