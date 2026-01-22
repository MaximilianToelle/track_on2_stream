import time

import torch
import torch.nn.functional as F
import kornia
from kornia.feature.responses import BlobDoGSingle

import cv2
import numpy as np


def not_used_cv2_sift_feature_coords(rgbs, segmentations, num_sift_features_per_view, device):
    # Obtain SIFT feature coordinates from rgbs inside segmentation_masks
    B = rgbs.shape[0]
    device = rgbs.device

    sift = cv2.SIFT_create()
    key_points_per_view = []
    min_num_sift_features_per_view = num_sift_features_per_view
    for b in range(B):
        rgb_np = rgbs[b].permute(1,2,0).cpu().numpy()
        seg_np = segmentations[b].cpu().numpy()
        key_points = sift.detect(rgb_np, seg_np)
        key_points_xy = torch.tensor(
            np.array([kp.pt for kp in key_points]), 
            dtype=torch.float32    
        ).to(device)
        key_points_per_view.append(key_points_xy) 

        num_key_points_xy = key_points_xy.shape[0]
        if num_key_points_xy < min_num_sift_features_per_view:
            min_num_sift_features_per_view = num_key_points_xy
    # Randomly sample min_num_sift_features_per_view per view
    coords = []
    for kp in key_points_per_view:
        num_kp = kp.shape[0]
        # randomly select min_num_sift_features_per_view indices
        sampled_idx = torch.randperm(num_kp, device=device)[:min_num_sift_features_per_view]
        coords.append(kp[sampled_idx])
    coords = torch.stack(coords, dim=0).round().to(torch.int32)   # int32 for indexing depth confidence
    return coords


def not_used_kornia_feature_coords(norm_grayscales, norm_segmentation_masks, num_sift_features_per_view):
    """
    Docstring for not_used_kornia_feature_coords
    
    :param norm_grayscales: [B, 1, H, W], values within [0,1], float32
    :param norm_segmentation_masks: [B, 1, H, W], values within [0,1], float32
    :param num_sift_features_per_view: int
    """

    detector = kornia.feature.ScaleSpaceDetector(
        num_features=num_sift_features_per_view, 
        resp_module=BlobDoGSingle(),
    )
        
    lafs, _ = detector(norm_grayscales, mask=norm_segmentation_masks)
    coords = kornia.feature.get_laf_center(lafs).round().to(torch.int32)   # [B,N,xy]
    return coords
        

class QueryManager:
    def __init__(
        self,
        num_camera_views, 
        num_queries_per_view,
        extrinsics, 
        intrinsics,
        resample_after_num_invisible_t = 4,
        resample_uncertainty_threshold = 0.1,
    ):  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.B = num_camera_views
        self.N = num_queries_per_view

        self.extrinsics = extrinsics
        self.intrinsics = intrinsics 

        self.resample_after_num_invisible_t = resample_after_num_invisible_t
        self.resample_uncertainty_threshold = resample_uncertainty_threshold

        self.padded_queries = torch.zeros((self.B, self.N, 2), dtype=torch.float32, device=self.device)
        self.resample_mask = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)

        self.invisibility_history = torch.zeros(
            (self.resample_after_num_invisible_t, self.B, self.N), 
            dtype=torch.bool,
            device=self.device,
        ) 

        self.initialized = False

    def update_queries(
        self,
        new_rgbs,                                           # [B, 3, H, W]
        new_segmentations,                                  # [B, H, W]
        new_depth_images,                                   # [B, H, W]
        new_depth_confidences,                              # [B, H, W]
        last_prediction = None,                             # [B, N, 2]
        last_post_processed_invisibility_mask = None,       # [B, N]
        last_uncertainty = None,                            # [B, N]
    ):     
        """
        Samples all queries during initial timestep. Otherwise, checks which tracked points have been invisible for  
        self.resample_after_num_invisible_t or which had uncertain predictions in the last timestep and selects those for resampling.
        
        Return: Padded tensor of new queries, resample mask 
        """

        if not self.initialized: 
            # NOTE: working with mask instead of indices to prevent different number of entries per batch element!
            self.resample_mask.fill_(True)
            last_visibility = None
            self.initialized = True
        else: 
            # update visibility
            self.invisibility_history[1:].copy_(self.invisibility_history[:-1].clone())
            self.invisibility_history[0].copy_(last_post_processed_invisibility_mask)

            # resample if invisible throughout history or too uncertain
            too_uncertain = last_uncertainty > self.resample_uncertainty_threshold                # [B, N]
            self.resample_mask.copy_(self.invisibility_history.all(dim=0) | too_uncertain)        # [B, N]
            # self.resample_mask.fill_(False)
            last_visibility = ~last_post_processed_invisibility_mask

        self.sample_new_queries(         
            new_rgbs,                          
            new_segmentations,             
            new_depth_images,                 
            new_depth_confidences,            
            last_prediction,
            last_visibility,
        )

        return self.padded_queries, self.resample_mask

    def sample_new_queries(
        self,
        rgbs,                           # [B, 3, H, W]
        segmentations,                  # [B, H, W]
        depth_images,                   # [B, H, W]
        depth_confidences,              # [B, H, W]
        last_prediction = None,         # [B, N, 2] with per-point x,y img coordinates
        last_visibility = None,         # [B, N]
        depth_confidence_threshold = 30,
        min_pixel_dist = 2,             # 2D convolution kernel radius
        voxel_size = 0.05,                                  
    ): 
        """
        Samples queries randomly inside segmentations. Filters coordinates based on foreground segmentation border, depth_confidences and, if not initial timestep, based on closeness to last_prediction.
        Projects filtered coordinates into 3D and does voxel hashing to eliminate too close coordinates and overlapping coordinates from different views. 
        Uniform sampling over remaining voxels to obtain the updated 2D queries.  
        
        Return: Padded tensor of new queries
        """

        B, _, H, W = rgbs.shape

        # create occupancy grid over batch elements (background is occupied)
        occupancy_grid = (~segmentations.to(torch.bool)).to(torch.float32)

        if last_prediction is not None and last_visibility is not None: 
            # mark currently tracked points as occupied
            # flatten B and N to update the grid in one shot
            batch_idx_flat = torch.arange(B, device=self.device).unsqueeze(1).expand(B, self.N).reshape(-1)
            last_visibility_flat = last_visibility.reshape(-1)
            batch_idx_visible_flat = batch_idx_flat[last_visibility_flat]
            visible_coords_flat = last_prediction.reshape(-1, 2)[last_visibility_flat]

            px = visible_coords_flat[:, 0].long().clamp(0, W - 1)
            py = visible_coords_flat[:, 1].long().clamp(0, H - 1)
            
            occupancy_grid[batch_idx_visible_flat, py, px] = 1.0
        
        # vectorized neighbor check (convolution)
        kernel_size = 2 * min_pixel_dist + 1
        kernel = torch.ones((B, 1, kernel_size, kernel_size), dtype=torch.float32, device=self.device)
        # move batch to channel dim: [1, B, H, W] for independent convolution
        neighbor_count = F.conv2d(
            occupancy_grid.unsqueeze(0),      
            kernel, 
            padding=kernel_size // 2, 
            groups=B
        ).squeeze(0)
        
        # Potential candidates: foreground, no neighbors, and good depth confidence
        candidate_mask = (neighbor_count == 0) & (depth_confidences <= depth_confidence_threshold)

        # TODO: Project candidates into 3D and filter wrt to overlapping view coordinates
        # TODO: Possibly we also need to change the number of queries per view to be a maximum value (smaller surfaces should not have as many queries as larger surfaces)

        # vectorized sampling via random priority
        # NOTE: since we can't easily torch.randperm with different counts per batch,
        # we assign a random value to every pixel and take the topk.
        random_priority = torch.rand((B, H, W), device=self.device)
        random_priority[~candidate_mask] = -1.0

        num_resample = self.resample_mask.sum(dim=1)      # [B]
        max_resample = num_resample.max()

        # Get topk random valid pixels for each batch element - topk works on a single dimension
        _, top_idx = torch.topk(random_priority.view(B, -1), k=max_resample, dim=1)     # [B, max_resample], [B, max_resample]

        # Convert flat indices back to (x, y)
        new_y = torch.div(top_idx, W, rounding_mode='trunc')
        new_x = top_idx % W
        sampled_coords = torch.stack([new_x, new_y], dim=-1).to(torch.float32)                    # [B, max_resample, 2]

        # map sampled coordinates back into a padded query buffer
        # [B, max_resample, 2] into [B, N, 2] where resample_mask is True (only [B, num_resample, 2] needed!)
        # treating resample mask as the scatter index
        destination_indices = self.resample_mask.nonzero(as_tuple=True)      # b_indices, n_indices
        # need a corresponding index for the sampled_coords
        # so far, we sampled "top max_resample" per batch element but only need "top num_to_resample"
        max_resample_indices = torch.arange(max_resample, device=self.device).unsqueeze(0).expand(B, -1)
        src_mask = max_resample_indices < num_resample.unsqueeze(1)
        
        self.padded_queries[destination_indices] = sampled_coords[src_mask]
