import time

import torch
import torch.nn.functional as F
        

class PointRegistry:
    def __init__(
        self,
        num_camera_views, 
        total_global_points,
        extrinsics_optical_to_world, 
        intrinsics,
        width,
        height,
        voxel_size,             
        resample_after_num_invisible_t,
        uncertainty_threshold,
        view_point_tracking_reset_threshold,
        device='cuda',
    ):  
        self.device = device
        self.B = num_camera_views
        self.N = total_global_points
        self.H, self.W = height, width
        self.voxel_size = voxel_size
        self.resample_after_num_invisible_t = resample_after_num_invisible_t
        self.resample_uncertainty_threshold = uncertainty_threshold  

        # --- Camera Setup ---
        self.K = intrinsics
        self.c2w = extrinsics_optical_to_world
        self.w2c = torch.linalg.inv(self.c2w)
        
        # --- Global State ---
        self.global_points_3d = torch.zeros((self.N, 3), device=device, dtype=torch.float32)
        self.global_invisibility_counter = torch.zeros(self.N, device=device, dtype=torch.long)
        self.global_slot_occupied = torch.zeros(self.N, device=device, dtype=torch.bool)
        self.global_just_resampled_mask = torch.ones(self.N, device=device, dtype=torch.bool)
        
        # --- Camera Tracking State ---
        self.view_point_seen = torch.zeros((self.B, self.N), device=device, dtype=torch.bool)
        self.view_point_invisibility_counter = torch.zeros((self.B, self.N), device=device, dtype=torch.long)
        self.view_point_tracking_reset_threshold = view_point_tracking_reset_threshold # num invisible timesteps in camera view until reset can be triggered (if projected point is valid)

    def generate_queries(
        self, 
        current_depths,
        current_segs,
    ):
        """
        Resamples dead global points or not occupied global slots. 
        Projects all 3D points to 2D. 
        For each camera view, filters out points that are out of view, occluded or have invalid depth. 
        Adds a per-view, per-point resampling state to global_just_resampled_mask IF projected point is valid and has not been seen before or was invisible for a long time (triggers memory reset inside tracking model) 
        
        Args:
            current_depths: [B, H, W] - Depth at Time T
            current_segs:   [B, H, W] - Segmentation at Time T
        Returns:
            uv: [B, N, 2] Pixel coordinates (Invalid ones set to -1000)
            resample_mask: [B, N]
        """

        # --- Resampling Global Points (Fill Empty/Dead Slots) ---
        # NOTE: Important to do at the beginning of timestep to directly register resampled points inside model
        dead_points = (self.global_invisibility_counter > self.resample_after_num_invisible_t)
        points_to_resample = dead_points | (~self.global_slot_occupied)
        
        self.global_just_resampled_mask[:] = False

        if points_to_resample.any():
            self._resample_slots(points_to_resample, current_depths, current_segs)

        # --- Project Global 3D -> 2D Pixel Coords ---
        uv, z_proj = self._project_world_to_pixels(self.global_points_3d)   # [B, N, 2], [B, N]

        # --- Geometric Validity Checks ---
        # Bounds check
        in_frame = (uv[..., 0] >= 0) & (uv[..., 0] < self.W - 1) & \
                   (uv[..., 1] >= 0) & (uv[..., 1] < self.H - 1) & \
                   (z_proj > 0.01) 

        # Occlusion check - we sample the ACTUAL depth map at the projected location.
        # Normalize to [-1, 1] for grid_sample
        u_norm = 2.0 * uv[..., 0] / (self.W - 1) - 1.0
        v_norm = 2.0 * uv[..., 1] / (self.H - 1) - 1.0
        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(2)   # [B, N, 1, 2]
        
        # Sample depth values at grid locations (Nearest neighbor to avoid blurring depth edges)
        z_sensor = F.grid_sample(
            current_depths.unsqueeze(1),        # [B, 1 (C), H_in, W_in]   
            grid,                               # [B, N, 1, 2]  -> pretend its an image of shape [B, H_out, W_out, 2]
            mode='nearest', 
            padding_mode='zeros',               # zero depth is invalid
            align_corners=True
        ).squeeze(1).squeeze(-1)                # [B, C, H_out, W_out] -> [B, N]

        valid_depth_mask = (z_sensor > 0.01) & (~torch.isnan(z_sensor)) & (~torch.isinf(z_sensor))
        
        # Threshold check: If the sensor sees a surface >1cm closer than our point, the point is blocked.
        occlusion_margin = 0.01 
        is_occluded = (z_sensor < (z_proj - occlusion_margin)) 
        
        valid_query = in_frame & (~is_occluded) & valid_depth_mask

        # --- Per-Projected-Point Reset Logic with Resampling Update ---
        # Check if we need to reset track of global point in specific camera view
        # NOTE: self.view_point_invisibility_counter and self.view_point_seen are updated based on prediction and post-processing inside fuse_and_update()
        is_long_term_lost = (self.view_point_invisibility_counter > self.view_point_tracking_reset_threshold)
        is_first_sighting = valid_query & (~self.view_point_seen)
        view_reset_tracking = valid_query & (is_long_term_lost | is_first_sighting)

        # Update resampling: Global resampling is done in all views, tracking reset is done per projected point
        global_resample = self.global_just_resampled_mask.unsqueeze(0).expand(self.B, -1)
        final_resample_mask = global_resample | view_reset_tracking

        # Hide invalid queries off-screen
        uv[~valid_query] = -1000.0 

        return uv, final_resample_mask

    def fuse_and_update(
        self, 
        pred_tracks, 
        pred_visibility, 
        pred_uncertainty, 
        current_depths, 
        current_segs
    ):
        """
        Fuses 2D tracks back into 3D and checking their validity
        """
        # 1. Unproject Tracks to 3D
        # pred_tracks_3d: [B, N, 3], valid_depth: [B, N]
        # NOTE: object occlusion is not checked again, relying on correct model visibility prediction 
        pred_tracks_3d, valid_depth = self._project_pixels_to_world(pred_tracks, current_depths)

        # 2. Tracked points need to be inside the object segmentation mask
        inside_seg = self._sample_values_at_coords(current_segs, pred_tracks) > 0

        # 3. Combine Checks - "who is currently tracking the point successfully"
        # Trusted (Valid) if: Visible + Valid Depth + Inside Segmentation + Low Uncertainty + Slot Occupied
        is_confident = pred_uncertainty < self.resample_uncertainty_threshold
        valid_mask = pred_visibility & valid_depth & inside_seg & is_confident & self.global_slot_occupied.unsqueeze(0)

        # Update Camera View State (used inside generate_queries())
        self.view_point_invisibility_counter[valid_mask] = 0
        self.view_point_invisibility_counter[~valid_mask] += 1
        self.view_point_seen = self.view_point_seen | valid_mask

        # 4. Weighted Average Fusion
        # NOTE: Model predicts uncertainties as low as 1e-5
        # We clamp to a feasible minimum such that no camera is dominating in weights e.g. 1000:1 
        min_uncertainty = 0.01 
        safe_uncertainty = torch.clamp(pred_uncertainty, min=min_uncertainty)
        weights = 1.0 / safe_uncertainty
        weights[~valid_mask] = 0.0  

        # normalize weights and avoid division by zero for points invisible in all camera views
        total_weight = weights.sum(dim=0, keepdim=True)  # [1, N]
        total_weight_safe = torch.clamp(total_weight, min=1e-6)
        norm_weights = weights / total_weight_safe 

        weighted_avg = (pred_tracks_3d * norm_weights.unsqueeze(-1)).sum(dim=0)

        # A point is "Visible" if at least one camera had a reliable view
        is_visible_now = (total_weight.squeeze(-1) > 0)
        
        # Update Position
        # TODO: Use 1-Euro Filter here!
        if is_visible_now.any():
            self.global_points_3d[is_visible_now] = weighted_avg[is_visible_now]

        # 5. Lifecycle Management
        # Increment invisible counter for global points NOT seen this timestep 
        self.global_invisibility_counter[~is_visible_now] += 1
        self.global_invisibility_counter[is_visible_now] = 0

        return self.global_points_3d, is_visible_now, self.global_just_resampled_mask

    def _project_world_to_pixels(self, points_world):
        # [N, 3] -> [B, N, 2]
        R = self.w2c[:, :3, :3]
        t = self.w2c[:, :3, 3]
        
        # ([B, 3, 3] @ [N, 3].T).permute -> [B, N, 3] + [B, 1, 3]
        points_cam = torch.matmul(R, points_world.T).permute(0, 2, 1) + t.unsqueeze(1)
        
        # 3D Points in camera coordinate system -> homogeneous coordinates
        points_2d_homogeneous = torch.einsum('bij,bnj->bni', self.K, points_cam)
        
        # Perspective Projection to get actual pixel values
        z = points_2d_homogeneous[..., 2]
        z_safe = torch.clamp(z, min=1e-6)       # avoid division by zero
        uv = points_2d_homogeneous[..., :2] / z_safe.unsqueeze(-1)
        
        return uv, z_safe

    def _project_pixels_to_world(
        self, 
        tracks_2d, 
        depths
    ):
        u_norm = 2.0 * tracks_2d[..., 0] / (self.W - 1) - 1.0
        v_norm = 2.0 * tracks_2d[..., 1] / (self.H - 1) - 1.0
        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(2) 
        
        # Do grid sampling to get smooth depth values (No sudden depth jumps due to float->int conversion)
        z = F.grid_sample(
            depths.unsqueeze(1),        # [B, 1 (C), H, W]  
            grid,                       # [B, N, 1, 2]  -> pretend it is an image [B, H_out, W_out, 2]
            mode='nearest', 
            padding_mode='zeros',       # zero depth is invalid
            align_corners=True,
        ).squeeze(1).squeeze(-1)        # [B, N]
        
        # CHECK: Valid Depth
        valid_z_mask = (z > 0.01) & (~torch.isnan(z)) & (~torch.isinf(z))
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        
        fx = self.K[:, 0, 0].unsqueeze(1)
        fy = self.K[:, 1, 1].unsqueeze(1)
        cx = self.K[:, 0, 2].unsqueeze(1)
        cy = self.K[:, 1, 2].unsqueeze(1)
        
        x_cam = (tracks_2d[..., 0] - cx) * z / fx
        y_cam = (tracks_2d[..., 1] - cy) * z / fy
        
        points_cam = torch.stack((x_cam, y_cam, z), dim=-1)
        R = self.c2w[:, :3, :3]
        t = self.c2w[:, :3, 3]
        points_world = torch.einsum('bij,bnj->bni', R, points_cam) + t.unsqueeze(1)
        
        return points_world, valid_z_mask

    def _sample_values_at_coords(self, values_2d, coords):
        """
        Samples 0 if coords are outside values_2d!
        """

        u_norm = 2.0 * coords[..., 0] / (self.W - 1) - 1.0
        v_norm = 2.0 * coords[..., 1] / (self.H - 1) - 1.0
        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(2)
        
        values = F.grid_sample(
            values_2d.unsqueeze(1).float(), 
            grid, 
            mode='nearest', 
            padding_mode='zeros',   
            align_corners=True,
        ).squeeze(1).squeeze(-1)
        
        return values        

    def _resample_slots(
        self, 
        resample_slots_mask, 
        depths, 
        segs
    ):
        num_needed = resample_slots_mask.sum().item()
        if num_needed == 0: return

        # Only sample at valid pixel locations
        valid_pixels = (segs > 0) & (depths > 0) & (~torch.isnan(depths)) & (~torch.isinf(depths))
        if not valid_pixels.any(): return 

        b_idx, y_idx, x_idx = torch.where(valid_pixels)
        
        # We downsample the number of candidates if there are too many -> have reliable computational speed
        max_candidates = 15000  
        if b_idx.shape[0] > max_candidates:
            perm = torch.randperm(b_idx.shape[0], device=self.device)[:max_candidates]
            b_idx, y_idx, x_idx = b_idx[perm], y_idx[perm], x_idx[perm]

        z_vals = depths[b_idx, y_idx, x_idx]
        
        fx = self.K[b_idx, 0, 0]
        fy = self.K[b_idx, 1, 1]
        cx = self.K[b_idx, 0, 2]
        cy = self.K[b_idx, 1, 2]
        
        x_cam = (x_idx - cx) * z_vals / fx
        y_cam = (y_idx - cy) * z_vals / fy
        cam_candidates = torch.stack([x_cam, y_cam, z_vals], dim=-1)
        
        R = self.c2w[b_idx, :3, :3]
        t = self.c2w[b_idx, :3, 3]
        world_candidates = torch.einsum('nij,nj->ni', R, cam_candidates) + t

        # Voxel Hashing -> Quantization, Duplicate Removal, Reconstruction to World Coordinates
        # A. Quantize candidates
        cand_voxel_idx = (world_candidates / self.voxel_size).long()
        
        # B. Quantize existing points
        existing_points = self.global_points_3d[self.global_slot_occupied]
        existing_voxel_idx = (existing_points / self.voxel_size).long()
        
        # C. Fast Set Difference via Hashing (PyTorch's set operations do not support multi-dimensional rows!)
        p1, p2, p3 = 73856093, 19349663, 83492791   # large prime numbers 
        def hash_voxels(v):
            return v[:, 0] * p1 + v[:, 1] * p2 + v[:, 2] * p3

        cand_keys = hash_voxels(cand_voxel_idx)
        existing_keys = hash_voxels(existing_voxel_idx)
        
        # Mask out candidates that already exist and only keep truly new voxels without duplicates
        is_new_voxel = ~torch.isin(cand_keys, existing_keys)
        
        # --- Goal: Keep only one representative world candidate per new voxel --- 
        # Filter down to only potential new points
        valid_candidates = world_candidates[is_new_voxel]
        valid_keys = cand_keys[is_new_voxel]
        
        if valid_candidates.shape[0] == 0: return

        # Sort by key to group candidates in the same voxel together
        # NOTE: we first shuffle to not have the same spatial order inside a voxel (later candidate selection should be random)
        perm_shuffle = torch.randperm(valid_candidates.shape[0], device=self.device)
        valid_keys = valid_keys[perm_shuffle]
        valid_candidates = valid_candidates[perm_shuffle]
        
        sort_idx = torch.argsort(valid_keys)         
        sorted_keys = valid_keys[sort_idx]
        sorted_candidates = valid_candidates[sort_idx]
        
        # Identify boundaries where the key changes
        # mask[0] is always True (first element is always unique so far)
        # mask[i] is True if key[i] != key[i-1]
        unique_mask = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=self.device)
        if sorted_keys.shape[0] > 1:
            unique_mask[1:] = (sorted_keys[1:] != sorted_keys[:-1])
        
        # Select representatives 
        new_points = sorted_candidates[unique_mask]

        if new_points.shape[0] > 0:
            perm = torch.randperm(new_points.shape[0], device=self.device)
            new_points = new_points[perm]
            
            count = min(num_needed, new_points.shape[0])
            slot_indices = torch.nonzero(resample_slots_mask, as_tuple=True)[0]
            
            # --- UPDATE Global State ---
            self.global_points_3d[slot_indices[:count]] = new_points[:count]
            self.global_slot_occupied[slot_indices[:count]] = True
            self.global_invisibility_counter[slot_indices[:count]] = 0
            self.global_just_resampled_mask[slot_indices[:count]] = True
