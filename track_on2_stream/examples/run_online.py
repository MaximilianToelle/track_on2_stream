import time

import torch

from ..model.trackon_predictor import StreamPredictor
from .query_manager import QueryManager 


class TrackOn2Stream:
    def __init__(
        self,
        model_args, 
        num_camera_views, 
        num_queries_per_view,
        extrinsics,
        intrinsics,
        checkpoint_path=None,
        support_grid_size=20,
    ):  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = StreamPredictor(
            model_args,
            checkpoint_path,
            support_grid_size,
        ).to(self.device).eval()

        self.query_manager = QueryManager(
            num_camera_views, 
            num_queries_per_view,
            extrinsics, 
            intrinsics,
        )

        self.last_prediction = None
        self.last_visibility = None
        self.last_uncertainty = None

    def process(
        self,
        new_rgbs,                           # [B, H, W, 3]
        new_segmentations,                  # [B, H, W]
        new_depth_images,                   # [B, H, W]
        new_depth_confidences,              # [B, H, W]
        last_post_processed_non_visibility_mask,
    ):  
    
        new_rgbs = new_rgbs.permute(0, 3, 1, 2).contiguous()        # [B, 3, H, W]
        padded_new_queries, resampled_mask = self.query_manager.update_queries(
            new_rgbs,
            new_segmentations,
            new_depth_images,
            new_depth_confidences,
            self.last_prediction,   # assumption: points do not move far in one step
            self.last_visibility, 
            self.last_uncertainty,
            last_post_processed_non_visibility_mask
        )

        point_prediction, visibility, uncertainty = self.predictor(
            new_rgbs, 
            padded_new_queries, 
            resampled_mask
        )

        self.last_prediction = point_prediction
        self.last_visibility = visibility
        self.last_uncertainty = uncertainty

        return point_prediction, visibility, uncertainty, resampled_mask
