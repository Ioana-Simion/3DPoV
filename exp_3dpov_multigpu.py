import argparse
import copy
from datetime import date
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import deepspeed  # <-- NEW: Import DeepSpeed

from clustering import PerDatasetClustering
from data_loader import PascalVOCDataModule, SamplingMode, VideoDataModule, SPairDataset, ScanNetPairsDataset, LimitedDataset
from eval_metrics import PredsmIoU
from models import CoTrackerFF, FeatureExtractor, FeatureForwarder, PipsFF
from my_utils import denormalize_video_cotracker_compatible, find_optimal_assignment, denormalize_video, overlay_video_cmap
from matplotlib.colors import ListedColormap
import torchvision.transforms as trn
from torchvision.ops import roi_align
from dino_vision_transformer import vit_small
from timm.models.vision_transformer import VisionTransformer

from evaluator import LinearFinetuneModule, KeypointMatchingModule, ScanNetCorrespondenceModule

from image_transformations import Compose, Resize
import video_transformations
import numpy as np
import random
from exp_3dpov import PoV3DBackbone
from my_utils import all_gather_concat
import torch.distributed as dist
from models import FixedMaskPatchDropout
# from cotracker.utils.visualizer import Visualizer

import wandb
import timm
import matplotlib.pyplot as plt
import io
from diffsort import DiffSortNet
import math
import gc

project_name = "3DPoV"
cmap = ListedColormap([
    '#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080',
    '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'
])

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

class PoV3D(PoV3DBackbone):
    def __init__(
        self, input_size, vit_model, num_prototypes=200, topk=5,
        context_frames=6, context_window=6, grid_size=32, logger=None,
        model_type='dino', latent_tracking=False, feature_upsampling='bilinear',
        sk_epsilon=0.05, sk_k=10, use_symmetric_loss=False,
        use_EMA_teacher=False, teacher_momentum=0.9, mask_ratio=0, teacher_eval=False,
        use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1, teacher_feature_upsampling='bilinear', crop_ratio=0.5, use_hardsoft =False, hard_steepness=100, soft_steepness=5, nmb_ref = 7, use_ref_per_frame=False,
    ):
        super(PoV3D, self).__init__(
            input_size, vit_model, num_prototypes, topk, context_frames,
            context_window, logger, model_type, mask_ratio, use_lora, lora_r, lora_alpha, lora_dropout
        )
        self.grid_size = grid_size
        self.FF = CoTrackerFF(
            self.eval_spatial_resolution, context_frames, context_window,
            topk=topk, grid_size=grid_size, feature_head=None
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.latent_tracking = latent_tracking
        self.feature_upsampling = feature_upsampling
        self.teacher_feature_upsampling = teacher_feature_upsampling
        self.sk_epsilon = sk_epsilon
        self.sk_k = sk_k
        self.use_symmetric_loss = use_symmetric_loss
        self.teacher_eval = teacher_eval
        self.use_EMA_teacher = use_EMA_teacher
        self.mask_ratio = mask_ratio
        self.teacher_momentum = None
        self.crop_ratio = crop_ratio
        self.use_hardsoft = use_hardsoft
        self.hard_steepness = hard_steepness
        self.soft_steepness = soft_steepness
        self.nmb_ref = nmb_ref
        self.use_ref_per_frame = use_ref_per_frame

        if self.use_EMA_teacher:
            self.teacher = copy.deepcopy(self.feature_extractor)
            if 'clip' in model_type:
                self.teacher.model.patch_drop = None
            else:
                self.teacher.model.patch_drop = torch.nn.Identity()
            
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher_momentum = teacher_momentum
            # Add teacher's MLP head and prototypes
            self.teacher_mlp_head = copy.deepcopy(self.mlp_head)
            for param in self.teacher_mlp_head.parameters():
                param.requires_grad = False
            self.teacher_prototypes = nn.Parameter(copy.deepcopy(self.prototypes.data), requires_grad=False)
            self.teacher_upsampler = copy.deepcopy(self.up_sampler)
            for param in self.teacher_upsampler.parameters():
                param.requires_grad = False

        sort_net = 'bitonic'

        patch_size = self.feature_extractor.model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        self.patch_size = patch_size
        raw_crop_size = input_size * self.crop_ratio
        self.crop_size = int(round(raw_crop_size / patch_size) * patch_size)
        spatial_res = self.crop_size // patch_size
        self.np = spatial_res ** 2
        print(f"Using {np} patches per frame with patch size {patch_size} and crop size {self.crop_size} and crop ratio {self.crop_ratio}")

        if self.use_hardsoft:
            self.soft_sorter = DiffSortNet(
                sorting_network_type=sort_net,
                size= self.np,
                steepness=self.soft_steepness
            )
            self.hard_sorter = DiffSortNet(
                sorting_network_type=sort_net,
                size= self.np,
                steepness=self.hard_steepness
            )
        else:
            # only use one sorter
            self.soft_sorter = DiffSortNet(
                sorting_network_type=sort_net,
                size= self.np,
                steepness=self.soft_steepness
            )

    def eval(self):
        """Override the eval method to exclude FF from training mode."""
        super().eval()
        if self.mask_ratio > 0:
            self.feature_extractor.model.patch_drop.training = False
        self.FF.eval()  # Set FF to evaluation mode
        return self
    
    def compute_perm_matrix_one_ref(self, query_feature, ref_feature, distance_type='cosine', hard_assignment=False):
        """
        Compute soft permutation matrix between query and a single reference feature set.
        
        Args:
            query_feature: [bs, np_q, dim] — e.g., from first or last frame
            ref_feature: [bs, np_r, dim] — one reference frame's patches
            distance_type: 'cosine' or 'euclidean'
            
        Returns:
            perm: [bs, np_q, np_r, np_r] — soft permutation matrix per sample
        """
        bs, np_q, dim = query_feature.shape
        np_r = ref_feature.shape[1]

        if distance_type == 'euclidean':
            dist = torch.cdist(query_feature, ref_feature, p=2.0)  # [bs, np_q, np_r]
        elif distance_type == 'cosine':
            dist = torch.einsum(
                'bnd,bmd->bnm',
                F.normalize(query_feature, dim=-1),
                F.normalize(ref_feature, dim=-1)
            )  # [bs, np_q, np_r]

        dist = dist.view(bs * np_q, np_r)
        if hard_assignment:
            _, perm = self.hard_sorter(dist)  # [bs * np_q, np_r, np_r]
        else:
            _, perm = self.soft_sorter(dist)  # [bs * np_q, np_r, np_r]

        perm = perm.view(bs, np_q, np_r, np_r).permute(0, 3, 2, 1)
    
        return perm

    def select_reference_tokens(self, feature_map, crop_size, patch_size, method="random"):
        """
        Crop a region from flattened ViT feature map post-forward.
        """
        bs, num_patches, dim = feature_map.shape
        grid_size = int(num_patches ** 0.5)  # assume square input
        H = W = grid_size
        feature_map = feature_map.view(bs, H, W, dim)

        crop_h = crop_w = crop_size // patch_size
        assert H >= crop_h and W >= crop_w, "Crop size too large for the feature map grid."

        if method == "center":
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            cropped = feature_map[:, top:top+crop_h, left:left+crop_w, :]  # [bs, crop_h, crop_w, dim]
            # if teacher_feat_map is not None:
            #     teacher_cropped = teacher_feat_map[:, top:top+crop_h, left:left+crop_w, :]
        elif method == "random":
            # Random top-left per sample in the batch
            tops = torch.randint(0, H - crop_h + 1, (bs,))
            lefts = torch.randint(0, W - crop_w + 1, (bs,))
            cropped = torch.stack([
                feature_map[i, top:top+crop_h, left:left+crop_w, :]
                for i, (top, left) in enumerate(zip(tops, lefts))
            ])  # [bs, crop_h, crop_w, dim]
        else:
            raise ValueError("Invalid crop method")
        
        return cropped.contiguous().view(bs, -1, dim) # [bs, crop_h * crop_w, dim]
    

    def train_step(self, datum, frames_to_compare=0):
        """
        One training step for 3DPov.
        This is what we will call inside our DeepSpeed engine for the forward pass.
        """

        # we are not using prototypes in this model, so we can skip this
        # self.normalize_prototypes()
        # if self.use_EMA_teacher:
        #     with torch.no_grad():
        #         w = self.teacher_prototypes.data.clone()
        #         w = F.normalize(w, dim=1, p=2)
        #         self.teacher_prototypes.copy_(w)    

        
        bs, nf, c, h, w = datum.shape
        denormalized_video = denormalize_video_cotracker_compatible(datum)
        if self.use_ref_per_frame:
            num_ref_frames = nf -1
        else:
            num_ref_frames = self.nmb_ref - 1
        #num_ref_frames = self.nmb_ref - 1 # we will sample num_ref_frames from other videos, and one intra-frame from the same video
        #num_ref_frames = nf -1
        # this way we have nmb_ref representing the actual references used
        all_frames = datum.flatten(0, 1)

        # Sample reference frames from other videos
        external_refs = []
        for i in range(bs):
            own_indices = list(range(i * nf, (i + 1) * nf))
            other_indices = list(set(range(bs * nf)) - set(own_indices))

            # Sample external frames
            if len(other_indices) < num_ref_frames:
                print(f"Warning: Not enough external frames to sample {num_ref_frames} references. Sampling from available {len(other_indices)} frames. datum shape is {datum.shape}")
                sampled_external = random.choices(other_indices, k=num_ref_frames)
            else:
                sampled_external = random.sample(other_indices, k=num_ref_frames)

            # Sample one frame from own video (exclude 0 and nf-1 if needed)
            avoid = [0, nf - 1]
            valid_intra = [j for j in own_indices if j % nf not in avoid]
            sampled_intra = random.sample(valid_intra, k=1) if valid_intra else [own_indices[0]]  # fallback

            all_sampled = sampled_external + sampled_intra  # length = num_ref_frames + 1
            external_refs.append(all_frames[all_sampled])   # [num_refs_total, C, H, W]

        num_ref_frames = num_ref_frames + 1 # +1 for the intra-frame

        external_refs = torch.stack(external_refs)  # [bs, num_ref_frames, C, H, W]
        flattened_refs = external_refs.view(bs * num_ref_frames, c, h, w)
        
        if datum.dtype == torch.float16:
            denormalized_video = denormalized_video.half()

        if self.mask_ratio > 0:
            B = bs
            L = self.feature_extractor.eval_spatial_resolution**2
            num_keep = max(0, int(L * (self.mask_ratio)))
            keep_indices = torch.argsort(torch.randn(B, L, device=datum.device), dim=-1)[:, num_keep: ]
            keep_indices = keep_indices.unsqueeze(1).repeat(1, nf, 1)
            keep_indices.requires_grad = False
            keep_indices = keep_indices.flatten(0, 1)
        else:
            keep_indices = None

        dataset_teacher_features = None
        if self.use_EMA_teacher:
            dataset_teacher_features, _ = self.teacher.forward_features(datum.flatten(0, 1), mask=None)
            dataset_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1), mask=keep_indices)
        else:
            dataset_features, _ = self.feature_extractor.forward_features(datum.flatten(0, 1), mask=keep_indices)

        ref_features, _ = self.feature_extractor.forward_features(flattened_refs, mask=None)

        # suppot for teacher processed features
        # if self.use_EMA_teacher:
        #     ref_features, _ = self.teacher.forward_features(flattened_refs, mask=None)
        # else:
        #     ref_features, _ = self.feature_extractor.forward_features(flattened_refs, mask=None)

        
        B, T, C, H, W = denormalized_video.shape
        with torch.no_grad():
            pred_tracks, pred_visibility = self.FF.forward(denormalized_video)

        pred_tracks.requires_grad = False
        pred_visibility.requires_grad = False
        _, _, dim = dataset_features.shape
        

        if self.latent_tracking:
            # New sampling approach
            reshaped_features = dataset_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
            reshaped_features = reshaped_features.permute(0, 3, 1, 2).contiguous()
            reshaped_features = reshaped_features.reshape(bs, nf, dim, self.spatial_resolution, self.spatial_resolution)

            if dataset_teacher_features is not None:
                reshaped_teacher_features = dataset_teacher_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
                reshaped_teacher_features = reshaped_teacher_features.permute(0, 3, 1, 2).contiguous()
                reshaped_teacher_features = reshaped_teacher_features.reshape(bs, nf, dim, self.spatial_resolution, self.spatial_resolution)

            pred_tracks = torch.clamp(pred_tracks, min=0, max=max(h-1, w-1))
            # Normalize pred_tracks to [-1, 1]
            pred_tracks_normalized = pred_tracks.float()
            pred_tracks_normalized[..., 0] = 2.0 * pred_tracks_normalized[..., 0] / (h - 1) - 1.0
            pred_tracks_normalized[..., 1] = 2.0 * pred_tracks_normalized[..., 1] / (w - 1) - 1.0

            if datum.dtype == torch.float16:
                pred_tracks_normalized = pred_tracks_normalized.half()

            grid = pred_tracks_normalized.reshape(bs, nf, -1, 1, 2)  # [bs, nf, num_points, 1, 2]

            # Sample features via grid_sample
            selected_features = []
            if dataset_teacher_features is not None:
                selected_teacher_features = []

            for t in range(nf):
                features_t = reshaped_features[:, t]  # [bs, dim, sr, sr]
                grid_t = grid[:, t]                  # [bs, num_points, 1, 2]
                sampled = F.grid_sample(
                    features_t, grid_t, mode='bilinear', align_corners=True
                )  # [bs, dim, num_points, 1]
                selected_features.append(sampled.squeeze(-1).permute(0, 2, 1))

                if dataset_teacher_features is not None:
                    teacher_features_t = reshaped_teacher_features[:, t]
                    sampled_teacher = F.grid_sample(
                        teacher_features_t, grid_t, mode='bilinear', align_corners=True
                    )
                    selected_teacher_features.append(sampled_teacher.squeeze(-1).permute(0, 2, 1))

            selected_features = torch.stack(selected_features, dim=1)  # [bs, nf, num_points, dim]
            if dataset_teacher_features is not None:
                selected_teacher_features = torch.stack(selected_teacher_features, dim=1)
        else:
            # Original sampling approach
            resized_reshaped_features = self._resize_features_orig_res(bs, nf, h, w, dataset_features, upsampling_mode=self.feature_upsampling)
            if dataset_teacher_features is not None:
                resized_reshaped_teacher_features = self._resize_features_orig_res(bs, nf, h, w, dataset_teacher_features, upsampling_mode=self.teacher_feature_upsampling)

            batch_idx = torch.arange(bs).view(bs, 1, 1).expand(-1, nf, pred_tracks.shape[2])
            time_idx = torch.arange(nf).view(1, nf, 1).expand(bs, -1, pred_tracks.shape[2])
            pred_tracks = torch.clamp(pred_tracks, min=0, max=max(h-1, w-1)).round().long()

            # pred_tracks = torch.randint(0, h, pred_tracks.shape)
            # (x,y) x-width y-height
            selected_features = resized_reshaped_features[
                batch_idx,
                time_idx,
                pred_tracks[..., 1].long(),  # height
                pred_tracks[..., 0].long(),  # width
            ]
            if dataset_teacher_features is not None:
                selected_teacher_features = resized_reshaped_teacher_features[
                    batch_idx,
                    time_idx,
                    pred_tracks[..., 1].long(),  # height
                    pred_tracks[..., 0].long(),  # width
                ]

        bs, nf, T, dim = selected_features.shape
        if self.mask_ratio > 0:
            # if dataset_teacher_features is not None:
            #     not_zero_rows = selected_teacher_features.sum(dim=-1) != 0
            # else:
            not_zero_rows = selected_features.sum(dim=-1) != 0
            not_zero_rows.requires_grad = False
        else:
            not_zero_rows = None


        projected_dataset_features = self.mlp_head(selected_features)
        if dataset_teacher_features is not None:
            projected_dataset_teacher_features = self.teacher_mlp_head(selected_teacher_features)
            projected_dim = projected_dataset_teacher_features.shape[-1]
            projected_dataset_teacher_features = projected_dataset_teacher_features.reshape(-1, projected_dim)
        projected_dim = projected_dataset_features.shape[-1]
        projected_dataset_features = projected_dataset_features.reshape(-1, projected_dim)
        normalized_projected_features = F.normalize(projected_dataset_features, dim=-1, p=2)
        normalized_projected_teacher_features = None
        if dataset_teacher_features is not None:
            normalized_projected_teacher_features = F.normalize(projected_dataset_teacher_features, dim=-1, p=2)

            
        ref_features = self.select_reference_tokens(ref_features, self.crop_size, self.patch_size)  # [bs * num_ref_frames, np_r, dim]
        ref_features = self.mlp_head(ref_features)
        ref_proj_dim = ref_features.shape[-1]
        ref_features = ref_features.reshape(-1, ref_proj_dim)
        ref_features = F.normalize(ref_features, dim=-1, p=2)  # Normalize reference features

        ref_features = ref_features.view(bs, num_ref_frames, -1, ref_features.shape[-1])  # [bs, nrf, np_r, dim]
        normalized_projected_features = normalized_projected_features.view(bs, nf, -1, normalized_projected_features.shape[-1])  # [bs, nf, np_q, dim]
        if normalized_projected_teacher_features is not None:
            normalized_projected_teacher_features = normalized_projected_teacher_features.view(bs, nf, -1, normalized_projected_teacher_features.shape[-1])

        vis0 = pred_visibility[:, 0]  
        total_loss = 0.0

        if frames_to_compare is None:
            frames_to_compare = 0

        # Clamp between 0 and nf-2 
        frames_to_compare = min(max(frames_to_compare, 0), nf - 2)

        # Compute first frame to compare
        # We always skip frame 0, so base start is 1 + frames_to_compare
        start_idx = 1 + frames_to_compare  # in [1 .. nf-1]

        total_loss = 0.0
        for i in range(start_idx, nf):
            vis_frame = pred_visibility[:, i]
            visibility_weights = vis0 * vis_frame
            if self.use_ref_per_frame:
                ref_feat = ref_features[:, i - 1].unsqueeze(1) # use one different reference per feat 
            else:
                ref_feat = ref_features

            frame_loss = self.compute_perm_subloss(
                ref_feat,
                normalized_projected_features[:, i],
                normalized_projected_teacher_features[:, 0] if self.use_EMA_teacher else normalized_projected_features[:, 0],
                visibility_weights=visibility_weights,
            )
            total_loss += frame_loss

        # Normalize by number of student frames actually compared
        num_compared = nf - start_idx  # e.g., if start_idx=1 -> nf-1 comparisons
        avg_loss = total_loss / num_compared
        # no need to average over ref features
        # if training in 1 ref per frame ablation
        if not self.use_ref_per_frame:
            avg_loss = avg_loss / ref_features.shape[1]  # Normalize over references
        avg_loss = avg_loss.mean()  # Average over batch

        torch.cuda.empty_cache()
        return avg_loss

    def compute_perm_subloss(self, ref_features, normalized_projected_features, normalized_projected_teacher_features = None, visibility_weights=None):
        loss = 0.0
        for i in range(ref_features.shape[1]):
            ref_feat = ref_features[:, i]  # [bs, np_r, dim]
            if self.use_hardsoft:
                first_perm = self.compute_perm_matrix_one_ref(
                    normalized_projected_features, ref_feat,
                    distance_type='cosine', hard_assignment=True
                )  # [bs, np_q, np_r]
            else:
                first_perm = self.compute_perm_matrix_one_ref(
                    normalized_projected_features, ref_feat,
                    distance_type='cosine', hard_assignment=False
                )  # [bs, np_q, np_r]
            if self.use_EMA_teacher:
                last_perm = self.compute_perm_matrix_one_ref(
                    normalized_projected_teacher_features, ref_feat,
                    distance_type='cosine', hard_assignment=False
                )
            else:
                last_perm = self.compute_perm_matrix_one_ref(
                    normalized_projected_features, ref_feat,
                    distance_type='cosine', hard_assignment=False
                )

            loss = -torch.sum(first_perm * torch.log(last_perm + 1e-6), dim=1)  # Loss over ranks
            
            loss = loss.mean(dim=1)  # [bs, np_q]
            
            if visibility_weights is not None:
                loss = (loss * visibility_weights).sum(dim=1) / (visibility_weights.sum(dim=1) + 1e-6)  # [bs]
            else:
                loss = loss.mean(dim=1)  # [bs]


            final_loss += loss
        return final_loss

    def _resize_features_orig_res(self, bs, nf, h, w, dataset_features, upsampling_mode='bilinear'):
        dim = dataset_features.size(-1)
        reshaped_features = dataset_features.reshape(bs*nf, self.spatial_resolution, self.spatial_resolution, dim)
        reshaped_features = reshaped_features.permute(0, 3, 1, 2).contiguous()
        resized_features = []
        for feature in reshaped_features:
            if upsampling_mode == 'bilinear':
                feature = F.interpolate(feature.unsqueeze(0), size=(h, w), mode="bilinear")
            elif upsampling_mode == 'nearest':
                feature = F.interpolate(feature.unsqueeze(0), size=(h, w), mode="nearest")
            elif upsampling_mode == "student_learnable":
                feature = self.up_sampler(feature.unsqueeze(0))
            elif upsampling_mode == "teacher_learnable":
                feature = self.teacher_upsampler(feature.unsqueeze(0))
            feature = feature.permute(0, 2, 3, 1).squeeze(0)
            resized_features.append(feature)
        resized_reshaped_features = torch.stack(resized_features, dim=0)
        resized_reshaped_features = resized_reshaped_features.reshape(bs, nf, h, w, dim)
        return resized_reshaped_features
    

    def update_teacher(self):
        """
        Update teacher model using exponential moving average of student parameters
        """
        if self.teacher_momentum is None:
            return
        with torch.no_grad():
            # Update feature extractor
            for param_t, param_s in zip(self.teacher.parameters(), self.feature_extractor.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
            # Update MLP head
            for param_t, param_s in zip(self.teacher_mlp_head.parameters(), self.mlp_head.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
            # Update prototypes
            self.teacher_prototypes.data = (self.teacher_momentum) * self.teacher_prototypes.data + (1 - self.teacher_momentum) * self.prototypes.data
            
            # Update up_sampler
            for param_t, param_s in zip(self.teacher_upsampler.parameters(), self.up_sampler.parameters()):
                param_t.data = (self.teacher_momentum) * param_t.data + (1 - self.teacher_momentum) * param_s.data
            
        self.teacher.eval()

class PoV3DTrainerDS:
    """
    A trainer class adapted for DeepSpeed. This replaces the manual optimizer usage
    with DeepSpeed engine calls (forward, backward, step).
    """
    def __init__(
        self, 
        training_dataloader, 
        test_dataloader, 
        hbird_dataset,
        num_epochs, 
        logger,
        ds_engine,
        spair_dataset,
        spair_val=False, 
        scannet_dataset = None,
        scannet_val=False,
        checkpoint_name='',
        use_scheduled_ranks=False,
        scheduled_step=False
    ):
        """
        Args:
          ds_engine: The DeepSpeed engine that wraps our model & optimizer.
        """
        # We'll store some references for convenience.
        self.training_dataloader = training_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.logger = logger
        self.hbird_dataset = hbird_dataset
        
        # The underlying original model is accessible via ds_engine.module
        # but for code clarity, let's keep this pointer:
        self.ds_engine = ds_engine
        self.model = ds_engine.module
        self.spair_val = spair_val
        self.scannet_val = scannet_val
        if self.spair_val:
            self.spair_dataset = spair_dataset
        if self.scannet_val:
            self.scannet_dataset = scannet_dataset

        self.best_recall = 0
        self.best_scannet_recall = 0 
        self.checkpoint_name = checkpoint_name
        self.use_scheduled_ranks = use_scheduled_ranks
        self.scheduled_step = scheduled_step
        self._current_step = None
        self.frames_to_compare = 0


    def train(self):
        """
        The main training loop with DeepSpeed: 
        We iterate over epochs and steps using `self.ds_engine`.
        """
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")
            # (Optional) you can call validate here at intervals
            self.train_one_epoch(epoch)


    def _print_gradient_info(self):
        for n, p in self.ds_engine.module.named_parameters():
            if p.requires_grad:
                print(f"\n{n}:")
                print(f"Grad exists: {p.grad is not None}")
                if p.grad is not None:
                    print(f"Grad norm: {p.grad.norm().item()}")

    def scheduled_steepness(self, step, cycle_length= 1000, min_s= 2.0, max_s=6.0) -> float:
        # Cosine restart every `cycle_length` steps
        cycle_pos = step % cycle_length
        return min_s + 0.5 * (max_s - min_s) * (1 + math.cos(math.pi * cycle_pos / cycle_length))
    
    def scheduled_k(self, global_step, min_k, max_k, warmup_epochs=8):
        total_steps = warmup_epochs * len(self.training_dataloader)
        progress = min(global_step / total_steps, 1.0)
        return int(min_k + progress * (max_k - min_k))


    def train_one_epoch(self, epoch):
        """
        One epoch of training using DeepSpeed, with periodic validation.
        """
        self.ds_engine.train()
        epoch_loss = 0
        is_main_process = dist.get_rank() == 0
        local_rank = self.ds_engine.local_rank 
        self.ds_engine.module.soft_sorter = self.ds_engine.module.soft_sorter.to(self.ds_engine.device)
        # Push sort ops to the right devices
        self.ds_engine.module.soft_sorter.sorting_network = [[op.to(self.ds_engine.device) for op in layer] for layer in self.ds_engine.module.soft_sorter.sorting_network]
        if self.ds_engine.module.use_hardsoft:
            self.ds_engine.module.hard_sorter = self.ds_engine.module.hard_sorter.to(self.ds_engine.device)
            # Push sort ops to the right devices
            self.ds_engine.module.hard_sorter.sorting_network = [[op.to(self.ds_engine.device) for op in layer] for layer in self.ds_engine.module.hard_sorter.sorting_network]
        
        # if self.scheduled_step:
        #     new_step = min(8 + 1 * epoch, 12) 
        #     self.training_dataloader.dataset.regular_step = new_step # reach inner dataset
        #     print(f"[Epoch {epoch}] Updated regular_step to {new_step}")
        
        
        if self.scheduled_step:
            # Increases every 8 epochs by 2
            increment = (epoch // 3) * 2
            new_step = min(6 + increment, 12)
            if self._current_step != new_step:
                if hasattr(self.training_dataloader.dataset, 'set_regular_step'):
                    self.training_dataloader.dataset.set_regular_step(new_step)
                else:
                    self.training_dataloader.dataset.regular_step = new_step
                print(f"[Epoch {epoch}] Updated regular_step to {new_step}")
                self._current_step = new_step
        for i, batch in enumerate(self.training_dataloader):
            global_step = epoch * len(self.training_dataloader) + i
            # new_steepness = self.scheduled_steepness(global_step, cycle_length=300, min_s=2.0, max_s=6.0)
            # self.ds_engine.module.soft_sorter.steepness = new_steepness

            datum = batch
            #annotations = annotations.squeeze(1).to(local_rank)
            datum = datum.squeeze(1).to(local_rank)
        
            if self.ds_engine.fp16_enabled():
                datum = datum.half()
            if datum.shape[0] == 1:
                print("[train] Skipping batch with only one valid sample")
                continue
           
            # Ensure the model is in training mode
            self.ds_engine.train()
            self.ds_engine.module.train()
            clustering_loss = self.ds_engine.module.train_step(datum, frames_to_compare=self.frames_to_compare)
            
            self.ds_engine.backward(clustering_loss)
            #self._print_gradient_info()
            self.ds_engine.step()
            if self.ds_engine.module.use_EMA_teacher:
                self.ds_engine.module.update_teacher()

            epoch_loss += clustering_loss.item()

            if is_main_process:
                self.logger.log({"3DPoV_loss": clustering_loss.item()})
                lr = self.ds_engine.get_lr()[0]
                self.logger.log({"lr": lr})

            print(f"Iteration: {i}, Loss: {clustering_loss.item()}")

            # Periodic validation
            num_itr = len(self.training_dataloader)
            
            if i% 218 == 0:
                self.validate(epoch)
            if i % 218 == 0:
                if self.scannet_val:
                    self.validate_scannet(epoch * num_itr + i)

            mixed = len(self.training_dataloader) // 2 +1
            if i % 218 == 0:
                self.save_model(epoch, i)
            if global_step % 1085 == 0:
                print(f"Frames to compare: {self.frames_to_compare} at global step {global_step}")
                self.frames_to_compare += 1
            if i % 218 == 0:
                self.validate_spair(epoch * num_itr + i)  


            torch.cuda.empty_cache()
        epoch_loss /= (i + 1)
        print(f"Epoch {epoch} Loss: {epoch_loss}")

    def save_model(self, epoch, iteration):
        
        checkpoint_dir = "/projects/0/prjs1400/checkpoints/3DPoV"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, f"model_iteration_{iteration}_epoch_{epoch}_{self.model.model_type}_{self.checkpoint_name}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved on iteration {iteration}:  at epoch {epoch}")

    def visualize_perm_matrices(self, last_perm_matrix, first_perm_matrix, ref_idx=0, rank_idx=0):
        """
        Visualizes:
        - First perm matrix (initial)
        - Ground-truth final matrix (directly from DiffSort)
        for the given reference index and rank.
        """
        is_main_process = dist.get_rank() == 0
        if not is_main_process:
            return

        # Extract [np_q, np_r] matrices for the selected reference and rank
        gt_final = last_perm_matrix[ref_idx].detach().cpu().numpy()
        init_final = first_perm_matrix[ref_idx].detach().cpu().numpy()

        # Normalize for colormap display
        def normalize(mat):
            return (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)

        gt_final = normalize(gt_final)
        init_final = normalize(init_final)

        # Plot both
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        axs[0].imshow(init_final, cmap='viridis')
        axs[0].set_title("Initial Perm Matrix")
        axs[1].imshow(gt_final, cmap='viridis')
        axs[1].set_title("Direct Final (DiffSort)")

        for ax in axs:
            ax.axis('off')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        compare_img = wandb.Image(img, caption="Initial vs Predicted Final")
        plt.close(fig)

        # Log
        self.logger.log({
            f"perm_matrix_comparison/ref_{ref_idx}/rank_{rank_idx}": compare_img
        })


    def validate_spair(self, epoch, threshold=0.10):
        """
        Distributed SPair validation with recall reporting for each viewpoint difference (0, 1, 2).
        """
        print('Starting SPair validation (per-vp_diff)')

        model = self.ds_engine.module
        model.eval()

        local_rank = self.ds_engine.local_rank
        global_rank = self.ds_engine.global_rank

        vp_datasets = self.spair_dataset

        per_vp_recalls = {}
        for vp, dataset in vp_datasets.items():
            if len(dataset) == 0:
                print(f"[Rank {global_rank}] No samples for vp_diff = {vp}")
                per_vp_recalls[vp] = -1.0
                continue

            evaluator = KeypointMatchingModule(
                model=model,
                dataset=dataset,
                device=f"cuda:{local_rank}",
                threshold=threshold
            )
            recall, _ = evaluator.evaluate_dataset(dataset, thresh=threshold, verbose=(global_rank == 0))

            # Gather recall across all processes
            recall_tensor = torch.tensor([recall], device=f"cuda:{local_rank}")
            gathered = [torch.zeros_like(recall_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, recall_tensor)
            avg_recall = torch.cat(gathered).mean().item()
            per_vp_recalls[vp] = avg_recall
            del evaluator
            torch.cuda.empty_cache()
            gc.collect()

        # Synchronize all ranks
        dist.barrier()
        vp_diff_labels = {
            0: "Small",
            1: "Medium",
            2: "Large"
        }
        # Rank 0: log results
        if global_rank == 0:
            for vp in [0, 1, 2]:
                label = vp_diff_labels[vp]
                recall = per_vp_recalls.get(vp, -1.0)
                if recall >= 0:
                    print(f"[Rank 0] SPair Recall@{threshold:.2f} | Viewpoint diff {label}: {recall:.2f}%")
                    self.logger.log({f"spair_recall_vp_{label.lower()}": recall})
                else:
                    print(f"[Rank 0] SPair Recall | Viewpoint diff {vp}: N/A")

            # Compute mean recall for valid view diffs
            valid_recalls = [r for r in per_vp_recalls.values() if r >= 0]
            if valid_recalls:
                avg_recall = sum(valid_recalls) / len(valid_recalls)
                print(f"[Rank 0] SPair Average Recall (vp0-2): {avg_recall:.2f}%")
                self.logger.log({"spair_keypoint_recall": avg_recall, "epoch": epoch})

                if avg_recall > self.best_recall:
                    self.best_recall = avg_recall
                    checkpoint_dir = "/projects/0/prjs1400/checkpoints/3DPoV"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"model_best_recall_epoch_{epoch}_{self.model.model_type}_{self.model.training_set}_{self.checkpoint_name}.pth"
                    )
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved with best recall: {avg_recall:.2f}% at epoch {epoch}")
        torch.cuda.empty_cache()
        # Ensure all processes synchronize
        dist.barrier()

    def validate_scannet(self, epoch, num_corr=200):
        print('Starting ScanNet validation')
        model = self.ds_engine.module
        model.eval()

        local_rank = self.ds_engine.local_rank
        global_rank = self.ds_engine.global_rank

        scannet_evaluator = ScanNetCorrespondenceModule(
            model=model,
            dataset=self.scannet_dataset,
            device=f"cuda:{local_rank}",
            num_corr=num_corr
        )
        print("Initialized ScanNet Correspondence Module")

        recall_10px = scannet_evaluator.evaluate()

        # Gather across all ranks
        recall_tensor = torch.tensor([recall_10px], device=f"cuda:{local_rank}")
        gathered_recalls = [torch.zeros_like(recall_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_recalls, recall_tensor)
        all_recalls = torch.cat(gathered_recalls).cpu().numpy()

        if global_rank == 0:
            avg_recall = all_recalls.mean()
            print(f"[Rank 0] ScanNet Correspondence Recall@10px: {avg_recall:.2f}%")
            self.logger.log({"scannet_10px_recall": avg_recall, "epoch": epoch})

            if avg_recall > self.best_scannet_recall:
                self.best_scannet_recall = avg_recall
                checkpoint_dir = "/projects/0/prjs1400/checkpoints/3DPoV"
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save_path = os.path.join(checkpoint_dir, f"model_best_scannet_epoch_{epoch}_{self.model.model_type}_{self.checkpoint_name}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved with best ScanNet recall: {self.best_scannet_recall:.2f}% at epoch {epoch}")

        dist.barrier()



    def validate(self, epoch, val_spatial_resolution=56):
        """
        Distributed validation that gathers all eval_targets and cluster_maps
        on every rank, then only rank 0 computes the final metric.
        """
        # Access the underlying model
        torch.cuda.empty_cache()
        model = self.ds_engine.module
        model.eval()
        
        # Current local rank / global rank
        local_rank = self.ds_engine.local_rank
        global_rank = self.ds_engine.global_rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # We'll store the local data here
        local_eval_features = []
        local_eval_targets = []

        model = self.ds_engine.module
        feature_spatial_resolution = model.feature_extractor.eval_spatial_resolution
        spatial_feature_dim = 50  # or model.get_dino_feature_spatial_dim()
        clustering_method = PerDatasetClustering(spatial_feature_dim, 21)
        overclustering_100_method = PerDatasetClustering(spatial_feature_dim, 100)
        #overclustering_300_method = PerDatasetClustering(spatial_feature_dim, 300)
        #overclustering_500_method = PerDatasetClustering(spatial_feature_dim, 500)
        
        with torch.no_grad():
            visualization_x = None
            for i, (x, y) in enumerate(self.test_dataloader):
                # Move input to local rank device
                img = x.to(local_rank)
                target = (y * 255).long().to(local_rank)

                if visualization_x is None and i == 0:
                    visualization_x = x  # keep a copy for logging on rank 0

                # 1) Get features
                spatial_features = model.validate_step(img)  # shape (B, np, dim)

                # 2) Resize your target if needed
                resized_target = F.interpolate(
                    target.float(), 
                    size=(val_spatial_resolution, val_spatial_resolution),
                    mode="nearest"
                ).long()

                # 3) Collect them in local lists
                local_eval_features.append(spatial_features)
                local_eval_targets.append(resized_target)

        # Concatenate along batch dimension on each GPU
        if local_eval_features:
            local_eval_features = torch.cat(local_eval_features, dim=0)  # shape [local_B, np, dim]
            local_eval_targets = torch.cat(local_eval_targets, dim=0)    # shape [local_B, H, W]
        else:
            # In case a rank has 0 samples (e.g., data not perfectly divisible):
            local_eval_features = torch.zeros((0, 50, 256), device=local_rank)  # or appropriate shape
            local_eval_targets = torch.zeros((0, val_spatial_resolution, val_spatial_resolution), device=local_rank)

        # 4) Gather all rank's features/targets to every GPU
        #    so that each GPU has the entire dataset.
        gathered_eval_features = all_gather_concat(local_eval_features)  # shape [total_B, np, dim]
        gathered_eval_targets = all_gather_concat(local_eval_targets)    # shape [total_B, H, W]

        # 5) On each GPU, produce cluster_maps
        B, npix, dim = gathered_eval_features.shape
        gathered_eval_features = gathered_eval_features.reshape(
            B,
            feature_spatial_resolution,
            feature_spatial_resolution,
            dim
        )
        gathered_eval_features = gathered_eval_features.permute(0, 3, 1, 2).contiguous()
        gathered_eval_features = F.interpolate(
            gathered_eval_features, 
            size=(val_spatial_resolution, val_spatial_resolution),
            mode="bilinear"
        )
        # shape now [B, dim, val_spatial_resolution, val_spatial_resolution]
        gathered_eval_features = gathered_eval_features.reshape(B, dim, -1).permute(0, 2, 1)
        # shape [B, HW, dim]
        gathered_eval_features = gathered_eval_features.detach().cpu().unsqueeze(1)
        # shape [B, 1, HW, dim]

        cluster_maps = clustering_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        #overclustering_300_maps = overclustering_300_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]
        
        overclustering_100_maps = overclustering_100_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        #overclustering_500_maps = overclustering_500_method.cluster(gathered_eval_features)
        # shape [B, val_spatial_resolution * val_spatial_resolution]

        cluster_maps = cluster_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        #overclustering_300_maps = overclustering_300_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        overclustering_100_maps = overclustering_100_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        #overclustering_500_maps = overclustering_500_maps.reshape(B, val_spatial_resolution, val_spatial_resolution).unsqueeze(1)
        # shape [B, 1, val_spatial_resolution, val_spatial_resolution]

        # 6) Now compute the metric only on rank 0 to avoid duplication
        if global_rank == 0:
            # valid_idx for ignoring 255
            # shape check: gathered_eval_targets is [B, H, W]
            # cluster_maps is [B, 1, H, W] => we can squeeze that to match
            
            valid_idx = gathered_eval_targets != 255
            valid_target = gathered_eval_targets[valid_idx].cpu()       # [some_size]
            valid_pred = cluster_maps[valid_idx.cpu()].cpu()         # [some_size]

            #valid_oc_pred = overclustering_300_maps[valid_idx.cpu()].cpu()
            valid_oc_pred_100 = overclustering_100_maps[valid_idx.cpu()].cpu()
            #valid_oc_pred_500 = overclustering_500_maps[valid_idx.cpu()].cpu()

            metric = PredsmIoU(21, 21)
            #overclustering_300_metric = PredsmIoU(300, 21)
            overclustering_100_metric = PredsmIoU(100, 21)
            #overclustering_500_metric = PredsmIoU(500, 21)
            metric.update(valid_target, valid_pred)
            #overclustering_300_metric.update(valid_target, valid_oc_pred)
            overclustering_100_metric.update(valid_target, valid_oc_pred_100)
            #overclustering_500_metric.update(valid_target, valid_oc_pred_500)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            #oc_300_jac, oc_300_tp, oc_300_fp, oc_300_fn, oc_300_reordered_preds, oc_300_matched_bg_clusters = overclustering_300_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
            oc_100_jac, oc_100_tp, oc_100_fp, oc_100_fn, oc_100_reordered_preds, oc_100_matched_bg_clusters = overclustering_100_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)
            #oc_500_jac, oc_500_tp, oc_500_fp, oc_500_fn, oc_500_reordered_preds, oc_500_matched_bg_clusters = overclustering_500_metric.compute(is_global_zero=True, many_to_one=True, precision_based=True)

            print(f"[Rank 0] Validation finished, global mIoU: {jac}")
            #print(f"[Rank 0] Validation finished, global overclustering mIoU: {oc_300_jac}")
            self.logger.log({"val_k=gt_miou": jac})
            #self.logger.log({"val_k=300_miou": oc_300_jac})
            self.logger.log({"val_k=100_miou": oc_100_jac})
            #self.logger.log({"val_k=500_miou": oc_500_jac})

        # If you want to ensure rank 0 finishes logging before others proceed, barrier:
        dist.barrier()
    
    def _log_video_to_wandb(self, video_tensor, fps=4, name="video"):
        """
        Log a video tensor to wandb.
        
        Args:
            video_tensor (torch.Tensor): Tensor of shape (T, 3, H, W) in uint8 format
            fps (int, optional): Frames per second. Defaults to 4.
            name (str, optional): Name of the video in wandb. Defaults to "video".
        """
        video = video_tensor.cpu().numpy()
        # video = np.transpose(video, (0, 2, 3, 1))
        self.logger.log({name: wandb.Video(video, fps=fps, format="mp4")})

    def log_cluster_maps(self, cluster_maps, denormalized_images):
        resized_denormalized_images= F.interpolate(denormalized_images, size=(cluster_maps.size(-2), cluster_maps.size(-1)), mode="bilinear")
        _, overlayed_video = overlay_video_cmap(cluster_maps.squeeze(1), resized_denormalized_images)
        self._log_images_to_wandb(overlayed_video, name="clustered_images")


    def _log_images_to_wandb(self, images, name="images"):
        self.logger.log({name: wandb.Image(images)})

    def _log_first_video_batch(self, datum, name="sampled_video_batch",  iteration = 0):
        bs, nf, c, h, w = datum.shape
        sample_batch = datum[:4]  # [4, nf, 3, h, w]

        # Undo normalization (TODO check)
        mean = torch.tensor([0.485, 0.456, 0.406], device=datum.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=datum.device).view(1, 1, 3, 1, 1)

        video = sample_batch * std + mean  # [4, nf, 3, h, w]
        video = torch.clamp(video * 255, 0, 255).to(torch.uint8)

        for idx, vid in enumerate(video):  # vid: [nf, 3, h, w]
            self._log_video_to_wandb(vid, fps=3, name=f"{name}_{idx}_{iteration}")

    def visualize_masked_patches(self, datum, name="masked_video", iteration = 0):
        """
        Visualize masked patches as black squares over video frames.

        Args:
            datum (Tensor): shape [B, T, 3, H, W]
            name (str): WandB video name
        """
        bs, nf, c, h, w = datum.shape
        sr = self.ds_engine.module.feature_extractor.eval_spatial_resolution
        patch_size = h // sr
        L = sr * sr
        device = datum.device

        # Take first few samples for visualization
        sample_batch = datum[:4]  # shape: [4, T, 3, H, W]
        sample_bs = sample_batch.shape[0]

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        video = sample_batch * std + mean
        video = torch.clamp(video * 255, 0, 255).to(torch.uint8).cpu()  # shape [4, T, 3, H, W]

        # Simulate keep indices
        num_keep = max(0, int(L * self.ds_engine.module.mask_ratio))
        keep_indices = torch.argsort(torch.randn(sample_bs, L, device=device), dim=-1)[:, num_keep:]
        keep_indices = keep_indices.unsqueeze(1).repeat(1, nf, 1).flatten(0, 1)

        # Create mask: True = masked, False = kept
        full_indices = torch.arange(L, device=device).unsqueeze(0).expand(sample_bs * nf, -1)
        mask = torch.ones_like(full_indices, dtype=torch.bool)
        mask.scatter_(1, keep_indices, False)

        # Draw black patches over masked regions
        for b in range(sample_bs):
            for t in range(nf):
                idx = b * nf + t
                frame = video[b, t]  # shape [3, H, W]
                masked_positions = mask[idx].nonzero(as_tuple=True)[0]
                for pos in masked_positions:
                    row = pos // sr
                    col = pos % sr
                    y0, y1 = row * patch_size, (row + 1) * patch_size
                    x0, x1 = col * patch_size, (col + 1) * patch_size
                    frame[:, y0:y1, x0:x1] = 0  

        # Log each video sample
        for b in range(sample_bs):
            vid = video[b]  # shape [T, 3, H, W]
            self._log_video_to_wandb(vid, fps=3, name=f"{name}_{b}_{iteration}")




    def log_tracked_frames(self, datum):
        """
        Process video frames and log the overlayed tracking visualization to wandb.
        
        Args:
            datum (torch.Tensor): Input video tensor
        """
        for d in datum[:10]:
            overlayed_video = self.ds_engine.module.log_tracked_frames(d.unsqueeze(0))  # B T 3 H W
            self._log_video_to_wandb(overlayed_video[0].to(torch.uint8), fps=4, name="overlayed_video")



def verify_arguments(args):

    if args.dataset == "epic_kitchens" and (args.frame_sampling_mode != "regular" or args.regular_step < 5):
        raise ValueError("Epic kitchens dataset is required for regular frame sampling mode and high regular step")
    
    if args.grid_size != 0 or args.latent_tracking:
        raise ValueError("Grid size and latent tracking are only supported for PoV3D")

    if args.frame_sampling_mode != 'regular' and args.regular_step > 0:
        raise ValueError("Regular step is only supported for regular frame sampling mode")
    
    if args.feature_upsampling != 'off' and args.latent_tracking:
        raise ValueError("Feature upsampling is not supported for latent tracking")
    
def run(args):
    """
    Main function adapted to launch DeepSpeed and do multi-GPU training.
    """
    verify_arguments(args)
    # ------------------------------
    # 1) SETUP WANDB and ARGS
    # ------------------------------
    device = f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda:0"
    config = vars(args)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    experiment_name = (
        f"grid:{args.grid_size}_latent:{args.latent_tracking}_"
        f"frame_sampling:{args.frame_sampling_mode}_model_type:{args.model_type}_batch_size:{args.batch_size}_"
        f"num_epochs:{args.num_epochs}_explaination:{args.explaination}_regular_step:{args.regular_step}_"
        f"num_clip_frames:{args.num_clip_frames}_num_clips:{args.num_clips}_num_gpus:{dist.get_world_size()}_num_epochs:{args.num_epochs}_feature_upsampling:{args.feature_upsampling}_num_prototypes:{args.num_prototypes}_sk_epsilon:{args.sk_epsilon}_sk_k:{args.sk_k}_use_symmetric_loss:{args.use_symmetric_loss}"
        f"_use_EMA_teacher:{args.use_EMA_teacher}_mask_ratio:{args.mask_ratio}_dataset:{args.dataset}_teacher_momentum:{args.teacher_momentum}_crop_scale:{args.crop_scale}_teacher_eval:{args.teacher_eval}"
        f"_use_lora:{args.use_lora}_lora_r:{args.lora_r}_lora_alpha:{args.lora_alpha}_lora_dropout:{args.lora_dropout}_mixed_datasets:{args.mixed_datasets}_sampling_ratios:{args.sampling_ratios}_teacher_feature_upsampling:{args.teacher_feature_upsampling}_use_hardsoft:{args.use_hardsoft}_learning_rate{args.learning_rate}"
    )   
    if dist.get_rank() == 0:
        logger = wandb.init(
            project=project_name,
            group=d1,
            mode=args.wandb_mode,
            job_type=args.job_type,
            config=config,
            name=experiment_name
        )
    else:
        logger = None

    # ------------------------------
    # 2) PREPARE DATA
    # ------------------------------
    rand_color_jitter = video_transformations.RandomApply(
        [video_transformations.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )],
        p=0.8
    )
    data_transform_list = [
        rand_color_jitter,
        video_transformations.RandomGrayscale(),
        video_transformations.RandomGaussianBlur()
    ]
    data_transform = video_transformations.Compose(data_transform_list)

    
    if args.crop_scale > 0:
        video_transform_list = [
            video_transformations.Resize((args.input_size, args.input_size)),
            video_transformations.RandomHorizontalFlip(),
            video_transformations.RandomResizedCrop((args.input_size, args.input_size), scale=(args.crop_scale, 1.0)),
            video_transformations.ClipToTensor(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
            )
        ]
    else:
        video_transform_list = [
            video_transformations.Resize((args.input_size, args.input_size)),
            video_transformations.RandomHorizontalFlip(),
            video_transformations.ClipToTensor(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
            )
        ]
        video_transform_list_crop = [
            video_transformations.CenterCrop((args.input_size, args.input_size)),
            video_transformations.RandomHorizontalFlip(),
            video_transformations.ClipToTensor(
                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
            )
        ]
        print("Using PoV3D transofrms")
   
    video_transform = video_transformations.Compose(video_transform_list)
    if video_transform_list_crop:
        video_transform_scene = video_transformations.Compose(video_transform_list_crop):
    transformations_dict = {
        "data_transforms": None,
        "target_transforms": None,
        "shared_transforms": video_transform,
        "scene_transforms" : video_transform_scene,
    }

    

    prefix = args.prefix_path
    if args.training_set == 'ytvos':
        data_path = os.path.join(prefix, "ytvos/train/JPEGImages")
        print(f"data_path : {data_path}")
        annotation_path = os.path.join(prefix, "ytvos/train/Annotations")
        print(f"annotation_path : {annotation_path}")
        meta_file_path = os.path.join(prefix, "ytvos/train/meta.json")
        print(f"meta file : {meta_file_path}")
        path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    elif args.training_set == 'co3d':
        data_path = os.path.join(prefix, "zips")
        annotation_path = os.path.join(prefix, "zips")
        meta_file_path = os.path.join(prefix, "zips")
        print(f"data_path CO3D: {data_path}")  
        path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    elif args.training_set == 'dl3dv':
        data_path = prefix
        annotation_path = prefix
        meta_file_path = prefix
        path_dict = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
    
    if args.frame_sampling_mode == 'uniform':
        sampling_mode = SamplingMode.UNIFORM
    elif args.frame_sampling_mode == 'dense':
        sampling_mode = SamplingMode.DENSE
    elif args.frame_sampling_mode == 'full':
        sampling_mode = SamplingMode.FULL
    else:
        sampling_mode = SamplingMode.Regular

    if args.training_set == 'ytvos':
        video_data_module = VideoDataModule("ytvos", path_dict, args.num_clips, args.num_clip_frames, sampling_mode, args.regular_step, args.batch_size, args.num_workers)
    elif args.training_set == 'ytvos_trj':
        video_data_module = VideoDataModule( #"ytvos_trj", path_dict, num_clips, num_clip_frames, sampling_mode, regular_step, batch_size, num_workers, grid_size=args.grid_size
            "ytvos_trj", path_dict,
            num_clips=args.num_clips,
            num_clip_frames=args.num_clip_frames,
            sampling_mode=sampling_mode,
            regular_step=args.regular_step,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            grid_size=args.grid_size
        )
    elif args.training_set == 'co3d':
        video_data_module = VideoDataModule("co3d", path_dict, args.num_clips, args.num_clip_frames, sampling_mode, args.regular_step, args.batch_size, args.num_workers, grid_size=args.grid_size)
    elif args.training_set == 'dl3dv':
        video_data_module = VideoDataModule("dl3dv", path_dict, args.num_clips, args.num_clip_frames, sampling_mode, args.regular_step, args.batch_size, args.num_workers)
    elif args.training_set == 'mixed':
        # safety checks around the mixed dataset args
        # if len(args.mixed_datasets) != len(args.sampling_ratios):
        #     raise ValueError("The number of datasets in --mixed_datasets must match the number of values in --sampling_ratios")
        if not (0.99 <= sum(args.sampling_ratios) <= 1.01):  
            raise ValueError("Sampling ratios must sum to 1.0")
        path_dicts = {}
        # hardcoded for now but there are too many arguments (ensure prefix arg is correct)
        # TODO Check paths!
        if 'ytvos' in args.mixed_datasets: 
            data_path = os.path.join(prefix, "ytvos/train/JPEGImages")
            annotation_path = os.path.join(prefix, "ytvos/train/Annotations")
            meta_file_path = os.path.join(prefix, "ytvos/train/meta.json")
            print(f'Using YTVOS in mixed datasets')
            path_dicts['ytvos'] = {"class_directory": data_path, "annotation_directory": annotation_path, "meta_file_path": meta_file_path}
        if 'ytvos_trj' in args.mixed_datasets:
            ytvos_trj_path = os.path.join(prefix, "dataset/all_frames/train_all_frames/JPEGImages/")
            path_dicts['ytvos_trj'] = {"class_directory": ytvos_trj_path, "annotation_directory": ytvos_trj_path, "meta_file_path": ytvos_trj_path}
        if 'co3d' in args.mixed_datasets:
            print(f'Using CO3D in mixed datasets')
            co3d_path = '/projects/2/managed_datasets/co3d/zips'
            path_dicts['co3d'] = {"class_directory": co3d_path, "annotation_directory": co3d_path, "meta_file_path": co3d_path} 
        if 'dl3dv' in args.mixed_datasets:
            print(f'Using DL3DV in mixed datasets')
            dl3dv_path = os.path.join(prefix, "dl3dv/")
            path_dicts['dl3dv'] = {"class_directory": dl3dv_path, "annotation_directory": dl3dv_path, "meta_file_path": dl3dv_path} 
        if 'kinetics' in args.mixed_datasets:
            kinetics_path = '/scratch-nvme/ml-datasets/kinetics/k700-2020'
            data_path = os.path.join(kinetics_path, 'train')
            annotation_path = os.path.join(kinetics_path, 'annotations')
            meta_file_path = os.path.join(kinetics_path, 'train')
            path_dicts['kinetics'] = {"class_directory": kinetics_path, "annotation_directory": kinetics_path, "meta_file_path": kinetics_path}
        print(f"Mixed datasets : {args.mixed_datasets}")
        if args.training_set == "mixed":
            video_data_module = VideoDataModule(
            "mixed",
            path_dicts,
            args.num_clips,
            args.num_clip_frames,
            sampling_mode,
            args.regular_step,
            args.batch_size,
            args.num_workers,
            mixed_datasets=args.mixed_datasets,
            sampling_ratios=args.sampling_ratios
            )

    video_data_module.setup(transformations_dict)
    video_data_module.make_data_loader()  # Creates .data_loader
    steps_per_epoch = 217
    # limit = steps_per_epoch * args.batch_size
    # video_data_module.dataset = LimitedDataset(video_data_module.dataset, limit=limit)
    video_data_module.make_data_loader()

    # Prepare model
    if args.model_type == 'dino':
        vit_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    if args.model_type == 'dino-b':
        #vit_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
        vit_model = torch.hub.load('facebookresearch/dino', 'dino_vitb16')
    elif args.model_type == 'dinov2':
        vit_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True, dynamic_img_size=True)
    elif args.model_type == 'dinov2-b':
        vit_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', img_size=args.input_size, pretrained=True, dynamic_img_size=True)
    elif args.model_type == 'registers':
        vit_model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', img_size=args.input_size, pretrained=True, dynamic_img_size=True)
    elif args.model_type == 'registers-b':
        vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        #vit_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', img_size=args.input_size, pretrained=True, dynamic_img_size=True)
    elif args.model_type == 'leopart':
        from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224
        prefix = os.environ.get("DATA_PREFIX")
        path_to_checkpoint = os.path.join(prefix, "leopart_knn/leopart_vits16.ckpt")
        vit_model = vit_small_patch16_224()  # or vit_base_patch8_224() if you want to use our larger model
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
    elif args.model_type == 'TimeT':
        from timm.models.vision_transformer import vit_small_patch16_224
        path_to_checkpoint = os.path.join(prefix, "paneco_models/TimeT.pth")
        vit_model = vit_small_patch16_224()
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict({".".join(k.split(".")[2:]): v for k, v in state_dict.items()}, strict=False)
    elif args.model_type == 'NeCo':
        path_to_checkpoint = os.path.join(prefix, "paneco_models/neco_on_dinov2r_vit14_model.ckpt")
        vit_model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        state_dict = torch.load(path_to_checkpoint)
        vit_model.load_state_dict(state_dict, strict=False)

    spair_dataset = None
    if args.spair_val:
        # print(f'spair_data_path: {args.spair_path}')
        # vp_diff = None
        # spair_dataset = SPairDataset(
        #     root=args.spair_path,
        #     split="test",
        #     use_bbox=False,
        #     image_size=224,
        #     image_mean="imagenet",
        #     class_name= None, # loop over classes in val if we want a per class recall
        #     num_instances=5000,
        #     vp_diff=vp_diff,
        # )
        # print(f'Length of SPair Dataset: {len(spair_dataset)}')
        
        spair_dataset = {
            0: SPairDataset(
                root=args.spair_path,
                split="test",
                use_bbox=False,
                image_size=224,
                image_mean="imagenet",
                class_name=None,
                num_instances=2200,  # Small vp_diff
                vp_diff=0,
            ),
            1: SPairDataset(
                root=args.spair_path,
                split="test",
                use_bbox=False,
                image_size=224,
                image_mean="imagenet",
                class_name=None,
                num_instances=1800,  # Medium vp_diff
                vp_diff=1,
            ),
            2: SPairDataset(
                root=args.spair_path,
                split="test",
                use_bbox=False,
                image_size=224,
                image_mean="imagenet",
                class_name=None,
                num_instances=1000,  # Large vp_diff
                vp_diff=2,
            ),
}

    scannet_dataset = None
    if args.scannet_val:
        scannet_dataset = ScanNetPairsDataset(
            root=args.scannet_path,
        )


    print(f"Using HARD SOFT is set to {args.use_hardsoft}")
    print(f"Using reference per frame is set to {args.use_ref_per_frame}")
    print(f"Using scheduled ranks is set to {args.use_scheduled_ranks}")

    
    patch_prediction_model = PoV3D(
        args.input_size, vit_model,
        logger=logger,
        model_type=args.model_type,
        grid_size=args.grid_size,
        latent_tracking=args.latent_tracking,
        feature_upsampling=args.feature_upsampling,
        num_prototypes=args.num_prototypes,
        use_symmetric_loss=args.use_symmetric_loss,
        use_EMA_teacher=args.use_EMA_teacher,
        teacher_momentum=args.teacher_momentum,
        mask_ratio=args.mask_ratio,
        teacher_eval=args.teacher_eval,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        teacher_feature_upsampling=args.teacher_feature_upsampling,
        use_hardsoft = args.use_hardsoft,
        hard_steepness = args.hard_steepness,
        soft_steepness = args.soft_steepness,
        nmb_ref = args.nmb_ref,
        crop_ratio= args.crop_ratio,
        use_ref_per_frame = args.use_ref_per_frame
    )



    # ------------------------------
    # 4) CALCULATE TOTAL STEPS
    # ------------------------------
    # The DataLoader is in video_data_module.data_loader
    steps_per_epoch = len(video_data_module.data_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    steps_per_epoch = 217 #align across runs
    # if 'co3d' in args.training_set:
    #     steps_per_epoch = 217 # try to match ytvos
    # elif 'mixed' in args.training_set:
    #     steps_per_epoch = steps_per_epoch // 2 + 1
    # If we do gradient_accumulation_steps > 1, we must account for that:
    gradient_accumulation_steps = 1  # or read from config
    total_training_steps = (steps_per_epoch * args.num_epochs) // gradient_accumulation_steps

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")


    # ------------------------------
    # 5) DEEPSPEED CONFIG (With Cosine LR & WD Scheduling)
    # ------------------------------
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "weight_decay": 1e-5  # Fixed weight decay value
            }
        },
        
        "fp16": {
            "enabled": False
        },
        
        "scheduler": {
            "type": "CosineAnnealingLR",
            "params": {
                "T_max": total_training_steps,
                "eta_min": 0,
            }
        },
}


    # ------------------------------
    # 4) Initialize DeepSpeed
    # ------------------------------
    # Note: We pass our dataset to `training_data` so that the engine knows how to distribute it.
    # If you'd rather do it manually, you can pass model_engine.train_batch(...) yourself.
    model_engine, optimizer, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=patch_prediction_model,
        model_parameters=patch_prediction_model.get_optimization_params(lr = args.learning_rate, weight_decay=1e-5),
        training_data=video_data_module.dataset,  # a PyTorch DataLoader
        config=ds_config
    )
    for i, group in enumerate(model_engine.optimizer.param_groups):
        print(f"Param Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, #params={len(group['params'])}")

    # Prepare val/test data
    image_val_transform = trn.Compose([
        trn.Resize((args.input_size, args.input_size)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    shared_val_transform = Compose([
        Resize(size=(args.input_size, args.input_size)),
    ])
    val_transforms = {
        "img": image_val_transform,
        "target": None,
        "shared": shared_val_transform
    }
    dataset = PascalVOCDataModule(batch_size=args.batch_size, train_transform=val_transforms, val_transform=val_transforms, test_transform=val_transforms, dir = args.pascal_path)

    dataset.setup()
    test_dataloader = dataset.get_test_dataloader()

    #ignore_index, hb_dataset = get_hb_dataset(args.hb_dataset_name, os.path.join(os.environ.get("DATA_PREFIX"), "VOCSegmentation_tiny"), 32, 224, 2)
    hb_dataset = None
    # Create our new trainer (DeepSpeed version)
    trainer = PoV3DTrainerDS(
        training_dataloader=training_dataloader,
        test_dataloader=test_dataloader,
        hbird_dataset=hb_dataset,
        num_epochs=args.num_epochs,
        logger=logger,
        ds_engine=model_engine,
        spair_dataset=spair_dataset, 
        spair_val=args.spair_val, 
        scannet_dataset = scannet_dataset,
        scannet_val = args.scannet_val,
        checkpoint_name=args.checkpoint_name,
        use_scheduled_ranks = args.use_scheduled_ranks,
        scheduled_step = args.scheduled_step
    )

    # ------------------------------
    # 5) TRAIN
    # ------------------------------
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DeepSpeed requires local_rank for multi-GPU usage
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DeepSpeed')

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument(
        '--frame_sampling_mode', type=str,
        choices=['uniform', 'dense', 'full', 'regular'], default='dense',
        help='Frame sampling mode: uniform, dense, full, or regular'
    )
    parser.add_argument(
        '--model_type', type=str,
        choices=['dino', 'leopart', 'dinov2', 'registers', 'TimeT', 'NeCo', 'dinov2-b', 'registers-b', 'dino-b'], default='dino',
        help='Select model type: dino or dinov2'
    )
    parser.add_argument(
        '--regular_step', type=int, default=1,
        help="Regular step for the video dataset"
    )
    parser.add_argument(
        '--job_type', type=str, default="debug_clustering_ytvos",
        help="Job type for wandb"
    )
    parser.add_argument("--explaination", type=str, default="full_frames")
    parser.add_argument('--grid_size', type=int, default=16, help='Grid size for the model')
    parser.add_argument('--latent_tracking', type=bool, default=False)
    parser.add_argument("--num_clip_frames", type=int, default=4)
    parser.add_argument("--num_clips", type=int, default=1)
    parser.add_argument("--num_prototypes", type=int, default=200)
    parser.add_argument("--sk_epsilon", type=float, default=0.05)
    parser.add_argument("--sk_k", type=int, default=10)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--feature_upsampling", type=str, choices=['bilinear', 'nearest', 'off'], default='bilinear')
    parser.add_argument("--use_symmetric_loss", type=bool, help="Use symmetric loss", default=False)
    parser.add_argument("--use_EMA_teacher", type=bool, help="Use EMA teacher", default=False)
    parser.add_argument("--teacher_momentum", type=float, help="Teacher momentum", default=0.99)
    parser.add_argument("--mask_ratio", type=float, help="Mask ratio", default=0)
    parser.add_argument("--hb_dataset_name", type=str, default="voc")
    

    #adapt based on dataset being used
    #parser.add_argument('--prefix_path', type=str, default="/projects/2/managed_datasets/co3d/")
    parser.add_argument('--prefix_path', type=str, default="/projects/0/prjs1400/")
    parser.add_argument('--pascal_path', type=str, default="/projects/0/prjs1400/pascal/VOCSegmentation")
    parser.add_argument('--spair_path', type=str, default="/home/isimion1/probe3d/data/SPair-71k")
    parser.add_argument('--scannet_path', type=str, default="/home/isimion1/probe3d/data/scannet_test_1500")
    parser.add_argument('--spair_val', type=float, default=False)
    parser.add_argument('--scannet_val', type=float, default=False)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--dataset", type=str, choices=['ytvos', 'epic_kitchens', 'mose', 'kinetics', 'co3d', 'dl3dv','mixed'], default="ytvos")
    parser.add_argument('--training_set', type=str, choices=['ytvos', 'co3d', 'dl3dv','mixed'], default='ytvos')
    parser.add_argument('--mixed_datasets', type=str, nargs='+', default=['co3d', 'ytvos'],
                    help="List of datasets to mix, e.g., '--mixed_datasets co3d ytvos_trj kinetics'")
    parser.add_argument('--sampling_ratios', type=float, nargs='+', default=[0.7, 0.3],
                    help="Sampling ratios for each dataset in '--mixed_datasets'. Must sum to 1.")
    parser.add_argument('--batch_frames', type=int, default=2, help='Number of reference frames to use from a batch')
    parser.add_argument('--crop_ratio', type=float, default=0.5, help='Crop ratio for the video frames')
    parser.add_argument('--use_lora', type=bool, default=False, help='Use LoRA fine-tuning instead of unfreezing layers')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout parameter')
    parser.add_argument("--crop_scale", type=float, help="Crop scale", default=0.0)
    parser.add_argument("--teacher_eval", type=bool, help="Teacher eval", default=False)
    parser.add_argument("--teacher_feature_upsampling", type=str, choices=['bilinear', 'nearest', 'off', 'teacher_learnable'], default='bilinear')
    parser.add_argument("--use_hardsoft", type=bool, default=False)
    parser.add_argument('--hard_steepness', type=int, default=100, help='Steepness for the hard storter')
    parser.add_argument('--soft_steepness', type=int, default=10, help='Steepness for the soft storter')
    parser.add_argument('--nmb_ref', type=int, default=7, help='Number of reference to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument("--use_ref_per_frame", type=bool, default=False)
    parser.add_argument("--use_scheduled_ranks", type=bool, default=False)
    parser.add_argument("--scheduled_step", type=bool, default=False)
    args = parser.parse_args()

    # This call sets up distributed training for DeepSpeed
    deepspeed.init_distributed()

    run(args)