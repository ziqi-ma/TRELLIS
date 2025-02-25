import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
from einops import rearrange, repeat
import flash_attn
from trellis.modules.sparse import SparseTensor
from trellis.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis.modules.attention.full_attn import scaled_dot_product_attention

class AttentionBaseSparse:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass
    
    def _sparse_scaled_dot_product_attention(self, qkv):
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        return s.replace(out)
    
    def sparse_scaled_dot_product_attention(self, qkv):
        out = self._sparse_scaled_dot_product_attention(qkv)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass
    
    def _scaled_dot_product_attention(self, q, k, v):
        return flash_attn.flash_attn_func(q, k, v)
    
    def scaled_dot_product_attention(self, q, k, v):
        out = self._scaled_dot_product_attention(q, k, v)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0



class MutualSelfAttentionControlSparse(AttentionBaseSparse):

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, total_layers=48):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            total_layers: layers of attn
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads):
        """
        Performing attention for a batch of queries, keys, and values
        q is T,h,dim_h where T is total length of all sequences
        k and v are first_len, h, dim_h
        """
        head_dim = q.shape[-1]
        total_l = q.shape[0]
        q = rearrange(q, "t h d -> h t d", h=num_heads)
        k = rearrange(k, "l h d -> h l d", h=num_heads)
        v = rearrange(v, "l h d -> h l d", h=num_heads)
        scale = 1.0 / (head_dim ** 0.5)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * scale # this will be h,t,l
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v) # this will be h t d
        out = rearrange(out, "h t d -> t h d", t = total_l)
        return out
    
    def _sparse_scaled_dot_product_attention(self, qkv): # this is without the incrememting step counter, wrapper is in base class
        if self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super()._sparse_scaled_dot_product_attention(qkv)
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        q, k, v = qkv.feats.unbind(dim=1) # each is T,h,dim_h
        h = qkv.feats.shape[-2]
        s = qkv
        firstlen = qkv.layout[0].stop
        out = self.attn_batch(q, k[:firstlen], v[:firstlen], h)
        return s.replace(out)
    

class MutualSelfAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, total_layers=48):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            total_layers: layers of attn
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads):
        """
        Performing attention for a batch of queries, keys, and values
        q is T,h,dim_h where T is total length of all sequences
        k and v are first_len, h, dim_h
        """
        head_dim = q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)

        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * scale
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out
    
    def _scaled_dot_product_attention(self, q, k, v): # this is without the incrememting step counter, wrapper is in base class
        if self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super()._scaled_dot_product_attention(q, k, v)
        num_heads = q.shape[2] #q,k,v (B, n, self.num_heads, -1)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> (b h) n d', h=num_heads), (q, k, v))
        # we only keep the k and v of the first instance which corresponds to the :h in dim 0
        out = self.attn_batch(q, k[:num_heads], v[:num_heads], num_heads)
        return out


def register_attention_editor_slat_flow(model, editor: AttentionBaseSparse):
    def ca_forward(self, place_in_unet):
        def forward(x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if not self._type == "self":
                # for now, don't modify cross attn
                q = self._linear(self.to_q, x)
                q = self._reshape_chs(q, (self.num_heads, -1))
                kv = self._linear(self.to_kv, context)
                kv = self._fused_pre(kv, num_fused=2)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=1)
                    k = self.k_rms_norm(k)
                    kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
                h = sparse_scaled_dot_product_attention(q, kv)
            
            else:
                qkv = self._linear(self.to_qkv, x)
                qkv = self._fused_pre(qkv, num_fused=3)
                if self.use_rope:
                    qkv = self._rope(qkv)
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=1)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
                #if self.attn_mode == "full": always full in our case
                h = editor.sparse_scaled_dot_product_attention(qkv)
            h = self._reshape_chs(h, (-1,))
            h = self._linear(self.to_out, h)
            return h

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'SparseMultiHeadAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count
    
    cross_att_count = register_editor(model.blocks, 0, "blocks")

    editor.num_att_layers = cross_att_count


def register_attention_editor_structure_gen(model, editor: AttentionBase):
    def ca_forward(self):
        def forward(x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            B, L, C = x.shape
            if not self._type == "self":
                # for now, don't modify cross attn
                Lkv = context.shape[1]
                q = self.to_q(x)
                kv = self.to_kv(context)
                q = q.reshape(B, L, self.num_heads, -1)
                kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(q, kv)
            
            else:
                qkv = self.to_qkv(x)
                qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
                if self.use_rope:
                    q, k, v = qkv.unbind(dim=2)
                    q, k = self.rope(q, k, indices)
                    qkv = torch.stack([q, k, v], dim=2)
                # only have full attn for now
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = editor.scaled_dot_product_attention(q, k, v)
                else:
                    # shouldn't hit here, will error out
                    h = editor.scaled_dot_product_attention(qkv)
                
            h = h.reshape(B, L, -1)
            h = self.to_out(h)
            return h

        return forward

    def register_editor(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'MultiHeadAttention':  # spatial Transformer layer
                net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count
    
    cross_att_count = register_editor(model.blocks, 0)

    editor.num_att_layers = cross_att_count
