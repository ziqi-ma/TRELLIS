from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
import torch.optim as optim
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult
import plotly.graph_objects as go
import wandb
from .samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from ..models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
from ..models.structured_latent_flow import SLatFlowModel
from ..models.sparse_structure_flow import SparseStructureFlowModel
from ..models.structured_latent_vae import SLatMeshDecoder, SLatGaussianDecoder, SLatRadianceFieldDecoder
from safetensors.torch import load_model
from torch.optim.lr_scheduler import CosineAnnealingLR
import imageio
from trellis.utils import render_utils, postprocessing_utils

def visualize_pts(points, colors, save_path=None, save_rendered_path=None):

    if save_path:
        np.save(f"{save_path}xyz.npy", points.cpu().numpy())
        np.save(f"{save_path}rgb.npy", colors.cpu().numpy())
    
    points = points.cpu().numpy()
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=(colors.cpu().numpy()*255).astype(int),  # Use RGB colors
            opacity=0.5
        ))])
    fig.update_layout(
        scene=dict(
            bgcolor='rgb(220, 220, 220)'  #bgcolor='rgb(255, 255, 255)'# Set the 3D scene background to light grey
        ),
        paper_bgcolor='rgb(220, 220, 220)' #paper_bgcolor='rgb(255, 255, 255)'## Set the overall figure background to light grey
    )
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),  # Adjust these values for your point cloud
            eye=dict(x=0, y=0, z=1.5),  # Increase the values to move further away
            center = dict(x=0,y=0,z=0)
        )
    )
    
    if save_rendered_path:
        fig.write_image(save_rendered_path)
    else:
        fig.show()


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device).float()
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionalEncodingXYZ(torch.nn.Module):
    def __init__(self, d_model=64*3):
        self.d_model = d_model
        super(PositionalEncodingXYZ, self).__init__()

    def forward(self, x): # x is n,3
        x_emb = get_1d_sincos_pos_embed_from_grid_torch(self.d_model//3, x[:,0])
        y_emb = get_1d_sincos_pos_embed_from_grid_torch(self.d_model//3, x[:,1])
        z_emb = get_1d_sincos_pos_embed_from_grid_torch(self.d_model//3, x[:,2])
        return torch.cat([x_emb, y_emb, z_emb],dim=1)
    
class TrellisEdit3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None
    ):
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self.cond_dim = 1024
        self.latent_proj_layer = nn.Linear(11,self.cond_dim) # 11 is concatenating latent and xyz
        self.emb_dim = 3*256 # 256 dim for each of x,y,z
        self.xyz_pos_emb = PositionalEncodingXYZ(d_model=self.emb_dim)
        self.pos_emb_proj = nn.Linear(self.emb_dim, self.cond_dim)

    @staticmethod
    def init_models() -> "TrellisEdit3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = TrellisEdit3DPipeline()

        pipeline.sparse_structure_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        pipeline.sparse_structure_sampler_params ={"steps": 25,
                                                   "cfg_strength": 5.0,
                                                   "cfg_interval": [0.5, 1.0],
                                                   "rescale_t": 3.0}

        pipeline.slat_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        pipeline.slat_sampler_params = {"steps": 25,
                                        "cfg_strength": 5.0,
                                        "cfg_interval": [0.5, 1.0],
                                        "rescale_t": 3.0}
        pipeline.models = {}
        pipeline.models['sparse_structure_encoder'] = SparseStructureEncoder(
            in_channels=1,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[32, 128, 512],
            use_fp16=True
        )
        load_model(pipeline.models['sparse_structure_encoder'], "/data/ziqi/checkpoints/trellis/ss_enc_conv3d_16l8_fp16.safetensors")
        
        pipeline.models["sparse_structure_decoder"] = SparseStructureDecoder(
            out_channels=1,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[512, 128, 32],
            use_fp16=True
        )
        load_model(pipeline.models['sparse_structure_decoder'], "/data/ziqi/checkpoints/trellis/ss_dec_conv3d_16l8_fp16.safetensors")
        
        pipeline.models["sparse_structure_flow_model"] = SparseStructureFlowModel(
            resolution=16,
            in_channels=8,
            out_channels=8,
            model_channels=1024,
            cond_channels=1024,
            num_blocks=24,
            num_heads=16,
            mlp_ratio=4,
            patch_size=1,
            pe_mode="ape",
            qk_rms_norm=True,
            use_fp16=False)
        
        pipeline.models["slat_flow_model"]=SLatFlowModel(
            resolution=64,
            in_channels=8,
            out_channels=8,
            model_channels=1024,
            cond_channels=1024,
            num_blocks=24,
            num_heads=16,
            mlp_ratio=4,
            patch_size=2,
            num_io_res_blocks=2,
            io_block_channels=[128],
            pe_mode="ape",
            qk_rms_norm=True,
            use_fp16=False
        )

        pipeline.models["slat_decoder_gs"]=SLatGaussianDecoder(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            use_fp16=True,
            representation_config={
                "lr": {
                    "_xyz": 1.0,
                    "_features_dc": 1.0,
                    "_opacity": 1.0,
                    "_scaling": 1.0,
                    "_rotation": 0.1
                },
                "perturb_offset": True,
                "voxel_size": 1.5,
                "num_gaussians": 32,
                "2d_filter_kernel_size": 0.1,
                "3d_filter_kernel_size": 9e-4,
                "scaling_bias": 4e-3,
                "opacity_bias": 0.1,
                "scaling_activation": "softplus"
            }
        )
        load_model(pipeline.models["slat_decoder_gs"], "/data/ziqi/checkpoints/trellis/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors")

        pipeline.models["slat_decoder_rf"] = SLatRadianceFieldDecoder(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            use_fp16=True,
            representation_config={
            "rank": 16,
            "dim": 8
            }
        )
        load_model(pipeline.models["slat_decoder_rf"], "/data/ziqi/checkpoints/trellis/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors")

        pipeline.models["slat_decoder_mesh"]=SLatMeshDecoder(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            use_fp16=True,
            representation_config={
                "use_color": True
            }
        )
        load_model(pipeline.models["slat_decoder_mesh"], "/data/ziqi/checkpoints/trellis/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors")

        pipeline.slat_normalization = {"mean": [
            -2.1687545776367188,
            -0.004347046371549368,
            -0.13352349400520325,
            -0.08418072760105133,
            -0.5271206498146057,
            0.7238689064979553,
            -1.1414450407028198,
            1.2039363384246826
            ],
            "std": [
                2.377650737762451,
                2.386378288269043,
                2.124418020248413,
                2.1748552322387695,
                2.663944721221924,
                2.371192216873169,
                2.6217446327209473,
                2.684523105621338
            ]}

        return pipeline
    

    @torch.no_grad()
    def encode_latent(self, coords, latent) -> torch.Tensor:
        """
        Encode the latent. coords is n_pts,3 and latent is n_pts,8 which is the trellis latent of the object to edit

        Args:
            coords: n_pts,3 of coordinate, xyz are 0-63
            latent: n_pts,8

        Returns:
            torch.Tensor: The encoded features.
        """
        coords = coords.to(self.device)
        latent = latent.to(self.device)
        coords_normalized = (coords/64)*2-1 # within -1 and 1
        all_feat = torch.cat([latent, coords_normalized], dim=1)
        feat_emb = self.latent_proj_layer(all_feat)

        pos_emb = self.xyz_pos_emb(coords)
        pos_emb = self.pos_emb_proj(pos_emb)
        features = feat_emb + pos_emb
        patchtokens = F.layer_norm(features, features.shape[-1:])
        
        # expected 1,...,1024
        return patchtokens.unsqueeze(0)
        
    def get_cond(self, coords, latent) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_latent(coords, latent)
        neg_cond = torch.zeros_like(cond).to(self.device)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        print(coords.shape)
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_slat_loss(
        self,
        cond: dict,
        coords: torch.Tensor,
        gt_slat: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        gt_slat = gt_slat.to(self.device)
        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)
        gt_slat_normalized = (gt_slat - mean) / std

        slat_loss = self.slat_sampler.sample_loss(
            flow_model,
            noise,
            gt_slat_normalized,
            **cond,
            **sampler_params,
            verbose=True)
        
        return slat_loss
    

    def sample_sparse_structure_loss(
        self,
        cond: dict,
        gt_coords: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        gt_coords = gt_coords.to(self.device)
        # transform gt_coords to the 64*64*64 shape
        gt_sparse = torch.zeros((num_samples, self.models['sparse_structure_encoder'].in_channels, 64,64,64)).cuda()
        zeros = torch.zeros(gt_sparse.shape[0]).int()
        gt_sparse[zeros,zeros,gt_coords[:,0],gt_coords[:,1],gt_coords[:,2]] = 1
        # encode ground truth and just calculate loss on latent space since the flow
        # model operates on this space
        gt_encoded = self.models['sparse_structure_encoder'](gt_sparse).float()

        z_s_loss = self.sparse_structure_sampler.sample_loss(
            flow_model,
            noise,
            gt_encoded,
            **cond,
            **sampler_params,
            verbose=True
        )
        
        return z_s_loss

    @torch.no_grad()
    def run(
        self,
        input_coords,
        input_latent,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
    ) -> dict:
        """
        Run the pipeline.

        Args:
            input_coords: input object's coordinates
            input_latent: input object's latent
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        cond = self.get_cond(input_coords, input_latent)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        #visualize_pts(coords[:,1:], torch.zeros(coords[:,1:].shape))
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        torch.save(slat, 'edited.pt')
        decoded = self.decode_slat(slat, formats)
        return decoded
    

    def get_latent_loss(
        self,
        input_coords,
        input_latent,
        output_coords,
        output_latent,
        num_samples: int = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {}
    ) -> dict:
        """
        Run the pipeline.

        Args:
            input_coords: input object's coordinates
            input_latent: input object's latent
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        cond = self.get_cond(input_coords, input_latent)
        #coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        gt_coords = torch.cat([torch.zeros(output_coords.shape[0],1), output_coords], dim=1).int()
        loss = self.sample_slat_loss(cond, gt_coords, output_latent, slat_sampler_params)
        return loss
    
    def get_sparse_structure_loss(
        self,
        input_coords,
        input_latent,
        output_coords,
        num_samples: int = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {}
    ) -> dict:
        """
        Run the pipeline.

        Args:
            input_coords: input object's coordinates
            input_latent: input object's latent
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        cond = self.get_cond(input_coords, input_latent)
        #gt_coords = torch.cat([torch.zeros(output_coords.shape[0],1), output_coords], dim=1).int()
        loss = self.sample_sparse_structure_loss(
            cond,
            output_coords,
            sampler_params=sparse_structure_sampler_params
        )
        return loss
    
    def get_all_loss(
        self,
        input_coords,
        input_latent,
        output_coords,
        output_latent,
        num_samples: int = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {}
    ) -> dict:
        """
        Run the pipeline.

        Args:
            input_coords: input object's coordinates
            input_latent: input object's latent
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        cond = self.get_cond(input_coords, input_latent)
        sparse_structure_loss = self.sample_sparse_structure_loss(
            cond,
            output_coords,
            sparse_structure_sampler_params
        )
        gt_coords = torch.cat([torch.zeros(output_coords.shape[0],1), output_coords], dim=1).int()
        latent_loss = self.sample_slat_loss(cond, gt_coords, output_latent, slat_sampler_params)
        return sparse_structure_loss, latent_loss


def train(lr, n_epoch, input_coords, input_latents, output_coords, output_latents, grad_clip=0.2):
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(project="trellis-overfit")
    
    #scaler = torch.cuda.amp.GradScaler()
    pipeline = TrellisEdit3DPipeline.init_models()
    pipeline.to(device) # this is moving all models
    # move our added layers to cuda, later should put these in a model and keep the pipeline just a pipeline
    pipeline.latent_proj_layer.to(device)
    pipeline.pos_emb_proj.to(device)
    pipeline.train()
    opt_ss = optim.Adam(list(pipeline.models['sparse_structure_flow_model'].parameters()) + list(pipeline.models["slat_flow_model"].parameters()) + list(pipeline.latent_proj_layer.parameters()) + list(pipeline.pos_emb_proj.parameters()), lr=lr)
    opt_latent = optim.Adam(list(pipeline.models["slat_flow_model"].parameters()), lr=lr)
    scheduler_ss = CosineAnnealingLR(opt_ss,n_epoch)
    scheduler_latent = CosineAnnealingLR(opt_latent,n_epoch)
    # we optimize the sparse generator and the latent generator at the same time, but keep the VAE fixed
    # i.e. use the existing learned latent space
    
    # we optimize in order, first just sparse generator + conditioning embedding
    iter = 0
    for epoch in range(n_epoch):
        epoch = epoch + 1
        #loss = pipeline.get_latent_loss(input_coords, input_latents, output_coords, output_latents)
        loss = pipeline.get_sparse_structure_loss(input_coords, input_latents, output_coords)
        opt_ss.zero_grad()
        #scaler.scale(loss).backward()
        loss.backward()
        
        #scaler.unscale_(opt)
        #torch.nn.utils.clip_grad_norm_(pipeline.models['slat_flow_model'].parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(pipeline.models['sparse_structure_flow_model'].parameters(), grad_clip)
        #scaler.step(opt)
        opt_ss.step()
        scheduler_ss.step()
        #scaler.update()
        values_to_log = {"ss iter":iter, "train ss loss": loss.item(), "ss lr": scheduler_ss.get_last_lr()[0]}
        wandb.log(values_to_log, step=iter, commit=True)
        iter += 1

    # we optimize in order, then we fit structured latent, fixing the conditional encoder
    # this could be changed later where we also optimize the conditional encoder here, but then we need to make sure
    # it works with the generator again
    wandb.define_metric("latent_step")
    wandb.define_metric("latent_loss", step_metric="latent_step")
    wandb.define_metric("latent_lr", step_metric="latent_step")
    iter = 0
    for epoch in range(n_epoch):
        epoch = epoch + 1
        loss = pipeline.get_latent_loss(input_coords, input_latents, output_coords, output_latents)
        opt_latent.zero_grad()
        #scaler.scale(loss).backward()
        loss.backward()
        
        #scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(pipeline.models['slat_flow_model'].parameters(), grad_clip)
        #scaler.step(opt)
        opt_latent.step()
        scheduler_latent.step()
        #scaler.update()
        values_to_log = {"latent_step":iter, "latent_loss": loss.item(), "latent_lr": scheduler_latent.get_last_lr()[0]}
        wandb.log(values_to_log, commit=True)
        iter += 1

    # now we finished optimizing both, use the weights to decode
    outputs = pipeline.run(
        input_coords,
        input_latents,
        num_samples= 1)
    
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("edited_gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave("edited_rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave("edited_mesh.mp4", video, fps=30)

    '''
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export("edited.glb")
    '''

    return loss.item()






