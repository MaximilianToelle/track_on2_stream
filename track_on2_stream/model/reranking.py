import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from ..utils.coord_utils import indices_to_coords
from .modules import DMSMHA_Block, MHA_Block


class Rerank_Module(nn.Module):
    def __init__(self, args, nhead=8):
        super().__init__()

        self.K = args.K
        self.D = args.D
        self.size = args.input_size
        self.stride = 4
        self.nhead = nhead

        self.H = self.size[0]
        self.W = self.size[1]
        self.Hf = self.H // self.stride
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)               

        self.reranking_layer_num = args.rerank_layer_num

        # === Top-K Point Decoder ===
        self.num_level = 4

        self.local_decoder = []
        for _ in range(self.reranking_layer_num):
            decoder_layer = DMSMHA_Block(self.D, nhead, self.num_level)
            self.local_decoder.append(decoder_layer)
        self.local_decoder = nn.ModuleList(self.local_decoder)
        # === === ===

        self.fusion = MHA_Block(self.D, nhead)

        self.fusion_layer = nn.Linear(2 * self.D, self.D)
        self.final_projection_layer = nn.Linear(2 * self.D, self.D)
        self.certainty_layer = nn.Linear(self.D, 1)
        self.score_layer = nn.Linear(self.D, 1)

        # Deformable inputs into buffers
        spatial_shapes = torch.tensor([(self.Hf, self.Wf), 
                                            (self.Hf // 2, self.Wf // 2), 
                                            (self.Hf // 4, self.Wf // 4), 
                                            (self.Hf // 8, self.Wf // 8)]) # (4, 2)
        self.register_buffer("spatial_shapes", spatial_shapes, persistent=False)

        
        start_levels = torch.tensor([0, 
                                     self.P, 
                                     self.P + self.P // 4,  
                                     self.P + self.P // 4 + self.P // 16]) # (4)
        self.register_buffer("start_levels", start_levels, persistent=False)

        scale = torch.tensor([self.size[1], self.size[0]], dtype=torch.float32).view(1, 1, 1, 2)
        self.register_buffer("scale_tensor", scale, persistent=False)


    def forward(self, q_t, f4, f8, f16, f32, c_t):
        # :args q_t: (B, N_t, D)
        # :args f4: (B, P, D)
        # :args f8: (B, P // 4, D)
        # :args f16: (B, P // 16, D)
        # :args f32: (B, P // 32, D)
        # :args c_t: (B, N_t, P), in size range

        # :return q_t: (B, N_t, D)
        # :return p_patch_top_k: (B, N_t, K, 2)
        # :return u_logit_top_k: (B, N_t, K)
        # :return s_logit_top_k: (B, N_t, K)

        B, N_t, D = q_t.shape
        _, P, D = f4.shape
        K = self.K
        device = q_t.device
        assert P == self.P, f"Expected P to be {self.P}, but got {P}"

        # === Top-k Indices ===
        top_k_indices = torch.topk(c_t, K, dim=-1)[1]                             # (B, N_t, K)
        p_patch_top_k = indices_to_coords(top_k_indices, self.size, self.stride)  # (B, N_t, K, 2), in [0, W] and [0, H] range
        # === === ===

        # === Local Decoder ===
        p_patch_top_k_norm = p_patch_top_k / self.scale_tensor       # (B, N_t, K, 2), in [0, 1] range
        p_patch_top_k_norm = torch.clamp(p_patch_top_k_norm, 0, 1)
        
        # reshape reference points
        # (B, N_t, K, 2) -> (B, N_t * K, 4, 2)
        p_patch_top_k_norm = p_patch_top_k_norm.view(B, N_t * K, 1, 2).expand(-1, -1, self.num_level, -1)    

        # concatenate multi-scale features  
        f_scales = torch.cat([f4, f8, f16, f32], dim=1)                                 # (B, P + P // 4 + P // 16 + P // 64, D)

        # reshape query
        q_top_k = q_t.view(B, N_t, 1, D).expand(-1, -1, K, -1).reshape(B, N_t * K, D)   # (B, N_t * K, D)
        q_t_expanded = q_t.unsqueeze(2).expand(-1, -1, K, -1)

        # === Run Local Decoder (Deformable Attention) ===
        # Batch size is preserved as B. Number of queries is N_t*K
        for layer in self.local_decoder:
            q_top_k = layer(
                q=q_top_k, 
                k=f_scales, 
                v=f_scales, 
                reference_points=p_patch_top_k_norm, 
                spatial_shapes=self.spatial_shapes, 
                start_levels=self.start_levels
            )
        q_top_k = q_top_k.view(B, N_t, K, D)
        q_top_k = torch.cat([q_top_k, q_t_expanded], dim=-1)                     # (B, N_t, K, 2 * D)
        q_top_k = self.fusion_layer(q_top_k)                                     # (B, N_t, K, D)
        # === === ===

        # === Fusion and Prediction ===
        u_logit_top_k = self.certainty_layer(q_top_k).squeeze(-1) # (B, N_t, K)
        s_logit_top_k = self.score_layer(q_top_k).squeeze(-1)     # (B, N_t, K)

        # Final Global Query Refinement 
        q_t_flat = q_t.view(B * N_t, 1, D)          
        q_top_k_flat = q_top_k.view(B * N_t, K, D)      
        
        # Attention: Fusion of the original query with its K candidates
        q_t_out = self.fusion(q_t_flat, q_top_k_flat, q_top_k_flat)                         # (B*N_t, 1, D)
        q_t_out = self.final_projection_layer(torch.cat([q_t_out, q_t_flat], dim=-1))       # (B*N_t, 1, D)
        q_t = q_t_out.view(B, N_t, D)  # (B, N_t, D)

        return q_t, p_patch_top_k, u_logit_top_k, s_logit_top_k




