import time

import torch

from ..model.trackon_predictor import StreamPredictor
from .point_registry import PointRegistry 


class TrackOn2Stream:
    def __init__(
        self,
        model_args, 
        num_camera_views, 
        num_global_points,
        voxel_size,
        resample_after_num_invisible_timesteps,
        resample_uncertainty_threshold,
        view_point_tracking_reset_threshold,
        extrinsics,                 # [B, 4, 4]
        intrinsics,                 # [B, 3, 3]
        height, 
        width,
        checkpoint_path=None,
        support_grid_size=20,
    ):  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_cameras = num_camera_views

        self.predictor = StreamPredictor(
            model_args,
            checkpoint_path,
            support_grid_size,
        ).to(self.device).eval()

        self.point_registry = PointRegistry(
            num_camera_views, 
            num_global_points, 
            extrinsics, 
            intrinsics, 
            width=width, 
            height=height,
            voxel_size=voxel_size,
            resample_after_num_invisible_t=resample_after_num_invisible_timesteps,
            uncertainty_threshold=resample_uncertainty_threshold,
            view_point_tracking_reset_threshold = view_point_tracking_reset_threshold,
            device=self.device,
        )

    def process(
        self,
        new_rgbs,                           # [B, H, W, 3]
        new_segmentations,                  # [B, H, W]
        new_depth_images,                   # [B, H, W]
    ):  
    
        new_rgbs = new_rgbs.permute(0, 3, 1, 2).contiguous()        # [B, 3, H, W]

        # start = time.time() 
        queries, resampled_mask_2d = self.point_registry.generate_queries(
            new_depth_images,
            new_segmentations
        )
        # print(f"Query Generation took: {time.time() - start}")
 
        # start = time.time()
        point_prediction, visibility, uncertainty = self.predictor(
            new_rgbs, 
            queries, 
            resampled_mask_2d
        )
        # print(f"Prediction took: {time.time() - start}")
        
        # start = time.time()
        points_3d, is_visible_mask_3d, resampled_mask_3d = self.point_registry.fuse_and_update(
            point_prediction,
            visibility,
            uncertainty,
            new_depth_images,
            new_segmentations
        )
        # print(f"Fuse and Update took: {time.time() - start}")

        is_visible_mask_3d.unsqueeze(0)

        return points_3d, is_visible_mask_3d, resampled_mask_3d
