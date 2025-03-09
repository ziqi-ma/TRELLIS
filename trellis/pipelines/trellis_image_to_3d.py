from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from torch.nn.utils.rnn import pad_sequence
from ..representations import Gaussian, Strivec, MeshExtractResult
from control.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlSparse, register_attention_editor_structure_gen, register_attention_editor_slat_flow
import random
import os
from trellis.utils.general_utils import visualize_pts
import torch.optim as optim
import torch
import clip
import numpy as np
from PIL import Image
from trellis.utils import render_utils
import imageio
from trellis.modules.norm import LayerNorm32
import wandb
NORMAL = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

def check_grad(module, grad_input, grad_output):
    # grad_input and grad_output are tuples
    # Check if any element is NaN or Inf
    for idx, g in enumerate(grad_input):
        if g is not None and torch.isnan(g).any():
            print(f"NaN in grad_input of {module} at index {idx}")
            print(grad_input)
            print(A)
        if g is not None and torch.isinf(g).any():
            print(f"Inf in grad_input of {module} at index {idx}")
            print(grad_input)
            print(A)
    for idx, g in enumerate(grad_output):
        if g is not None and torch.isnan(g).any():
            print(f"NaN in grad_input of {module} at index {idx}")
            print(grad_input)
            print(A)
        if g is not None and torch.isinf(g).any():
            print(f"Inf in grad_input of {module} at index {idx}")
            print(grad_input)
            print(A)

def check_forward(module, input, output):
    # If it's a SparseTensor, you might need to check output.feats
    feats = input[0].feats if hasattr(input[0], 'feats') else input[0]
    if torch.isnan(feats).any() or torch.isinf(feats).any():
        print(f"NaN/Inf detected in forward input of {module.__class__.__name__}")
        print(input[0])
        print(a)



class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
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
    
    def sample_sparse_structure_with_logprobs(self, cond, num_samples, sampler_params, eta):
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
        
        z_s, timesteps, latents, logprobs = self.sparse_structure_sampler.sample_noised_with_logprobs(
            flow_model,
            noise,
            **cond,
            eta=eta,
            **sampler_params,
            verbose=True
        )
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        return coords, timesteps, latents, logprobs # timesteps and latents are length n_steps+1, logprobs is length n_steps-1
    
    def sample_sparse_structure_consistent(
        self,
        cond,
        starting_step,
        starting_layer,
        stop_layer,
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
        stack_noise = torch.cat([noise, noise],dim=0)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        controller = MutualSelfAttentionControl(starting_step, starting_layer, total_layers=stop_layer)
        register_attention_editor_structure_gen(flow_model, controller)

        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            stack_noise,
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
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        if NORMAL:
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True
            ).samples
        else:
            slat = self.slat_sampler.sample_noised(
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
    
    def sample_slat_with_logprobs(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params,
        steps,
        eta
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
        sampler_params["steps"] = steps

        slat, timesteps, latents, logprobs = self.slat_sampler.sample_noised_with_logprobs(
            flow_model,
            noise,
            **cond,
            eta=eta,
            **sampler_params,
            verbose=True
        )

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, timesteps, latents, logprobs
    
    def sample_slat_consistent(
        self,
        cond0: dict,
        cond1: dict,
        coords0: torch.Tensor,
        coords1: torch.Tensor,
        starting_step,
        starting_layer,
        stop_layer,
        sampler_params: dict = {}
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
        
        # pack them in sequence
        coords1[:,0] = 1
        coords = torch.cat([coords0, coords1], dim=0)

        # stack conds
        cond_all = {}
        for k in cond0:
            cond_all[k] = torch.cat([cond0[k], cond1[k]], dim=0)

        noise_feats = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)

        noise = sp.SparseTensor(
            feats=noise_feats,
            coords=coords,
        )

        controller = MutualSelfAttentionControlSparse(starting_step, starting_layer, total_layers=stop_layer)
        register_attention_editor_slat_flow(flow_model, controller)

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond_all,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat
    
    def sample_slat_consistent(
        self,
        cond:dict,
        coords: torch.Tensor,
        starting_step,
        starting_layer,
        stop_layer,
        sampler_params: dict = {}
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

        noise_feats = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)

        noise = sp.SparseTensor(
            feats=noise_feats,
            coords=coords,
        )

        controller = MutualSelfAttentionControlSparse(starting_step, starting_layer, total_layers=stop_layer)
        register_attention_editor_slat_flow(flow_model, controller)

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

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        #minz = coords[:,3].min()
        #coords = coords[coords[:,3]>=minz+2]
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    

    @torch.no_grad()
    def run_consistent_tolatnet(
        self,
        images: List[Image.Image],
        starting_step,
        starting_layer, 
        stop_layer,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond0 = self.get_cond([images[0]])
        cond1 = self.get_cond([images[1]])
        
        set_seed(seed)
        coords0 = self.sample_sparse_structure(cond0, num_samples, sparse_structure_sampler_params)
        set_seed(seed)
        coords1 = self.sample_sparse_structure(cond1, num_samples, sparse_structure_sampler_params)
        set_seed(seed)
        slat = self.sample_slat_consistent(cond0, cond1, coords0, coords1, starting_step, starting_layer, stop_layer, slat_sampler_params)

        return self.decode_slat(slat, formats)
    

    @torch.no_grad()
    def run_consistent_all(
        self,
        images: List[Image.Image],
        starting_step_ss,
        starting_layer_ss, 
        stop_layer_ss,
        starting_step_l,
        starting_layer_l, 
        stop_layer_l,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond0 = self.get_cond([images[0]])
        cond1 = self.get_cond([images[1]])
         # stack conds
        cond_all = {}
        for k in cond0:
            cond_all[k] = torch.cat([cond0[k], cond1[k]], dim=0)
        
        set_seed(seed)
        coords = self.sample_sparse_structure_consistent(
            cond_all, starting_step_ss, starting_layer_ss, stop_layer_ss, **sparse_structure_sampler_params)
        
        visualize_pts(coords[coords[:,0]==0][:,1:], torch.zeros(coords[coords[:,0]==0][:,1:].shape))
        visualize_pts(coords[coords[:,0]==1][:,1:], torch.zeros(coords[coords[:,1]==0][:,1:].shape))
        
        slat = self.sample_slat_consistent(cond_all, coords, starting_step_l, starting_layer_l, stop_layer_l, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    @torch.no_grad()
    def run_with_ss_logprobs(
        self,
        cond,
        eta,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        
        torch.manual_seed(seed)
        coords, ss_timesteps, ss_latents, ss_logprobs = self.sample_sparse_structure_with_logprobs(cond, num_samples, sparse_structure_sampler_params, eta)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        res = self.decode_slat(slat, formats)
        return res, ss_timesteps, ss_latents, ss_logprobs
    
    @torch.no_grad()
    def run_with_slat_logprobs(
        self,
        cond,
        steps,
        eta,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat, slat_timesteps, slat_latents, slat_logprobs = self.sample_slat_with_logprobs(cond, coords, slat_sampler_params, steps, eta)
        res = self.decode_slat(slat, formats)
        return res, slat_timesteps, slat_latents, slat_logprobs
    
    @torch.no_grad()
    def get_reward_slat_text(self, slat_out, edit_prompt):
        # for now only use gaussian
        # and use CLIP similarity
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        views = render_utils.render_video(slat_out['gaussian'][0])['color'] # this is list of numpys
        img_input = [preprocess(Image.fromarray(img)).unsqueeze(0).to(device) for img in views]
        img_input = torch.cat(img_input)
        text_input = clip.tokenize([edit_prompt]).to(device)
        image_features = model.encode_image(img_input)
        text_features = model.encode_text(text_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # k, dim
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 1,dim
        similarity_score = (image_features @ text_features.T).cpu().numpy() # k,1
        mean_score = similarity_score.mean()
        return mean_score

    def train_ss_gen_with_reward(
        self,
        n_epochs: int, # n_epochs is how many times to sample noise and get "old" path and reward
        n_inner_train_epochs: int, #n_inner_train_epochs is how many times to update the model parameter - we essentially reuse the sample trajectory and only update model weights so log_prob will change
        lr,
        edit_prompt: str,
        image: Image.Image,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        adv_clip_max = 5,
        clip_range = 1e-4,
        eta=0.2
        ):

        optimizer = optim.Adam(self.models['sparse_structure_flow_model'].parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])

            #################### SAMPLING ####################
            self.models['sparse_structure_flow_model'].eval()

            samples = []
            # for now, we just have a single sample
            # EXTENSION: possible to change this to sampling multiple noises
            # ss latents should include the initial noise so size of time_steps+1!
            # so is ss_timesteps
            slat_out, ss_timesteps, ss_latents, ss_log_probs = self.run_with_ss_logprobs(
                cond,
                seed = seed + epoch,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                slat_sampler_params = slat_sampler_params,
                formats = formats)
            # log_probs should be a list of size n_timesteps-1
            # latents and timesteps should be a list of size n_timesteps+1
            
            ss_latents = torch.stack(ss_latents) # (num_steps+1, dim_latent)
            ss_log_probs = torch.stack(ss_log_probs) # (num_steps-1,)
            ss_timesteps = torch.stack(ss_timesteps) # (num_steps+1,)
                
            rewards = self.get_reward_slat_text(slat_out, edit_prompt) # a number

            samples.append(
                {
                    "ss_timesteps": ss_timesteps,
                    "ss_latents": ss_latents[:-1],  # each entry is the latent before timestep t
                    "ss_next_latents": ss_latents[1:],  # each entry is the latent after timestep t
                    "log_probs": ss_log_probs,
                    "rewards": rewards,
                }
            )

            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

            # TODO: right now we only have one reward, cannot do mean and std
            advantages  = rewards
            #advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            bs, num_timesteps = samples["timesteps"].shape # bs = 1 as of now
            num_timesteps -= 1
            print(bs)
            print(num_timesteps)

            #################### TRAINING ####################
            for inner_epoch in range(n_inner_train_epochs):
                # TODO: shuffle along batch dim if applicable, rn our bs =1

                # dict of lists -> list of dicts for easier iteration
                samples_batched = [
                    dict(zip(samples, x)) for x in zip(*samples.values())
                ]
                print(samples_batched)

                # train
                self.models['sparse_structure_flow_model'].train()

                #info = defaultdict(list)
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training"
                ):

                    for j in tqdm(
                        range(num_timesteps-1),
                        desc="Timestep"
                    ):
                        
                        ss_log_prob = self.sparse_structure_sampler.sample_onestep_logprob(
                            self.models['sparse_structure_flow_model'],
                            cond,
                            x_t = sample["latents"][:, j],
                            x_t_prev = sample["next_latents"][:, j],
                            t = sample["timesteps"][j],
                            t_prev =  sample["timesteps"][j+1],
                            eta = eta,
                            **sparse_structure_sampler_params)

                        # this needs to first predict noise then get logprob ddim
                        # look at ddim_step_with_logprob implementation, prob needs to carry 
                        # gradient of model
                    
                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -adv_clip_max,
                            adv_clip_max,
                        )

                        ratio = torch.exp(ss_log_prob - sample["log_probs"][:, j])
                        print(ratio)
                        unclipped_loss = -advantages * ratio
                        print(unclipped_loss)
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - clip_range,
                            1.0 + clip_range,
                        )
                        print(clipped_loss)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        '''
                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)
                        '''

                        # backward pass
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
        # save weight
        torch.save(self.models['sparse_structure_flow_model'], '/data/ziqi/training_checkpts/trellisrl/ss_gen_weights.pth')
        
        # perform rendering with a normal run now
        outputs = self.run(image,seed=seed)
        
        # Render the outputs
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(f"out/new_gs.mp4", video, fps=30)
        return
    
    def train_slat_with_reward(
        self,
        n_epochs: int, # n_epochs is how many times to sample noise and get "old" path and reward
        n_inner_train_epochs: int, #n_inner_train_epochs is how many times to update the model parameter - we essentially reuse the sample trajectory and only update model weights so log_prob will change
        lr,
        edit_prompt: str,
        image: Image.Image,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        adv_clip_max = 5,
        clip_range = 1e-4,
        steps=100,
        eta=0.2,
        inner_batch_size = 50
        ):
        save_path = f"{edit_prompt}s{steps}ib{inner_batch_size}"
        os.makedirs(f"out/{save_path}", exist_ok=True)
        #torch.autograd.set_detect_anomaly(True)
        # first load 
        #self.models['slat_flow_model'] = torch.load('/data/ziqi/training_checkpts/trellisrl/slat_weights.pth')
        self.models["slat_flow_model"] = self.models["slat_flow_model"].float()
        self.models["slat_flow_model"].convert_to_fp32()
        self.models["slat_flow_model"].dtype=torch.float32
        # flaot16 was causing so many issues!!!

        optimizer = optim.Adam(self.models["slat_flow_model"].parameters(), lr=lr)

        wandb.init(
            project="rl",
            name=f"{edit_prompt}s{steps}ib{inner_batch_size}"
        )

        iter = 0
        
        for epoch in range(n_epochs):
            
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])

            #################### SAMPLING ####################
            self.models["slat_flow_model"].eval()

            samples = []
            # for now, we just have a single sample
            # EXTENSION: possible to change this to sampling multiple noises
            # ss latents should include the initial noise so size of time_steps+1!
            # so is ss_timesteps
            slat_out, slat_timesteps, slat_latents, slat_log_probs = self.run_with_slat_logprobs(
                cond,
                steps=steps,
                eta=eta,
                seed = seed,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                slat_sampler_params = slat_sampler_params,
                formats = formats)
            # log_probs should be a list of size n_timesteps-1
            # latents and timesteps should be a list of size n_timesteps+1
            
            slat_log_probs = torch.stack(slat_log_probs) # (num_steps-1,)
            slat_timesteps = torch.tensor(np.array(slat_timesteps)) # (num_steps+1,)
                
            rewards = self.get_reward_slat_text(slat_out, edit_prompt) # a number
            print(f"reward: {rewards}")
            # write to reward
            with open(f'out/{save_path}/rewards', 'a') as file:
                file.write(f"{rewards}\n")
            

            samples.append(
                {
                    #"cond": cond['cond'], EXTENSION: when doing multiple samples, need to pass in cond
                    "timesteps": slat_timesteps,
                    "latents": slat_latents[:-1],  # list of SparseTensor, each entry is the latent before timestep t
                    "next_latents": slat_latents[1:],  # list of SparseTensor,each entry is the latent after timestep t
                    "log_probs": slat_log_probs,
                    "advantages": torch.tensor([rewards]), # one sample, no mean and std to work with
                }
            )

            #samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()} EXTENSION: do this when you have a batch

            # TODO: right now we only have one reward, cannot do mean and std
            advantages  = rewards
            #advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            num_timesteps = samples[0]["timesteps"].shape[0] # bs = 1 as of now EXTENSION this will be bs,num_steps
            num_timesteps -= 1

            #################### TRAINING ####################

            # train
            self.models['slat_flow_model'].train()
            '''
            for name, module in self.models["slat_flow_model"].named_modules():
                # Register a hook on each module
                module.register_full_backward_hook(check_grad)
                if isinstance(module, LayerNorm32):
                    module.register_forward_hook(check_forward)

            for name, param in self.models['slat_flow_model'].named_parameters():
                def check_param_grad(grad, param_name=name):
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"NaN/Inf in param {param_name} grad")
                    return grad  # Must return the grad to avoid modifying it

                if param.requires_grad:
                    param.register_hook(check_param_grad)
            '''
            onestep_sampler_params = {**self.slat_sampler_params, **slat_sampler_params}
            del onestep_sampler_params['steps']
            del onestep_sampler_params['rescale_t']
            #delete steps because below we only use it for one step
            for inner_epoch in range(n_inner_train_epochs):
                # TODO: shuffle along batch dim if applicable, rn our bs =1

                # dict of lists -> list of dicts for easier iteration
                #samples_batched = [
                    #dict(zip(samples, x)) for x in zip(*samples.values())
                #] EXTENSION: do this when batching multiple samples
                samples_batched = samples

                #info = defaultdict(list)
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training"
                ):
                    # in each inner epoch, randomly sample inner_batch_size timesteps and form a minibatch
                    t_samples = np.random.choice(num_timesteps-1, size=inner_batch_size, replace=False)
                    losses = []
                    for j in t_samples:#range(num_timesteps-1),:
                        set_seed(seed+epoch+i+j)
                        slat_log_prob = self.slat_sampler.sample_onestep_logprob(
                            self.models['slat_flow_model'],
                            x_t = sample["latents"][j], # EXTENSION, :,j so first dim is batch
                            x_t_prev = sample["next_latents"][j], # EXTENSION, :,j so first dim is batch
                            t = sample["timesteps"][j],
                            t_prev =  sample["timesteps"][j+1],
                            eta = eta,
                            **cond, # EXTENSION: When doing multiple samples, need to take samples['cond']
                            **onestep_sampler_params)

                        # this needs to first predict noise then get logprob ddim
                        # look at ddim_step_with_logprob implementation, prob needs to carry 
                        # gradient of model
                    
                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -adv_clip_max,
                            adv_clip_max,
                        )
                        #print(slat_log_prob)
                        #print(sample["log_probs"][j])

                        ratio = torch.exp(slat_log_prob - sample["log_probs"][j]).cpu() # EXTENSION, :,j so first dim is batch
                        #print(ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - clip_range,
                            1.0 + clip_range,
                        )
                       
                        curloss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        wandb.log({
                            "iter": iter+1,
                            "loss": curloss.item()
                        })
                        #print(curloss)
                        losses.append(curloss.item())
                        iter += 1

                        curloss.backward()
                        
                        '''
                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)
                        '''
                    print(losses)

                    wandb.log({
                            "iter": iter+1,
                            "loss": np.array(losses).mean()
                        })
                    
                    # step after the whole batch
                    optimizer.step()
                    optimizer.zero_grad()

                outputs = self.run(image,seed=seed)
                rewards = self.get_reward_slat_text(outputs, edit_prompt) # a number
                print(f"reward: {rewards}")
                wandb.log({
                    "iter": iter,
                    "reward": rewards
                    })
                # write to reward
                with open(f'out/{save_path}/rewards', 'a') as file:
                    file.write(f"{rewards}\n")
                self.models['slat_flow_model'].train()

            # Render the outputs
            if epoch == 0 or (epoch+1) % 5 == 0:
                video = render_utils.render_video(outputs['gaussian'][0])['color']
                imageio.mimsave(f"out/{save_path}/epoch{epoch}_gs.mp4", video, fps=30)

        # save weight
        torch.save(self.models['slat_flow_model'], '/data/ziqi/training_checkpts/trellisrl/slat_weights.pth')
        
        # perform rendering with a normal run now
        outputs = self.run(image,seed=seed)
        
        # Render the outputs
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(f"outnew_gs.mp4", video, fps=30)
        return
    
