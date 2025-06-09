import os
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules.sparse.basic import SparseTensor
import numpy as np
from safetensors.torch import load_model
from trellis.models.structured_latent_vae import SLatGaussianDecoder
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.


def load_gs_decoder(gs_decoder_save_path):
    gs_decoder = SLatGaussianDecoder(
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
    load_model(gs_decoder, gs_decoder_save_path)
    gs_decoder = gs_decoder.cuda()
    return gs_decoder

def load_sparse_tensor(in_path, eps=1, mode="l1"):

    slat_dense = torch.load(in_path)

    if mode == "l1":
        l1_sum = torch.abs(slat_dense).sum(dim=-1)
        nonzero_idxs = l1_sum > eps
        coords = (nonzero_idxs*1).nonzero(as_tuple=False)  # shape: (n, 3)
        features = slat_dense[nonzero_idxs]
    elif mode == "any":
        feature_lessthaneps = (torch.abs(slat_dense) < eps)*1
        feature_zeroany = (feature_lessthaneps).sum(dim=-1) # this is nonzero if any dimension <1
        feature_nonzero = (feature_zeroany == 0)*1
        coords = feature_nonzero.nonzero(as_tuple=False)  # shape: (n, 3)
        features = slat_dense[coords[:,0],coords[:,1], coords[:,2]]
    
    coords = torch.cat([torch.zeros(coords.shape[0],1),coords], dim=1).int()

    slat_sparse = SparseTensor(feats=features, coords=coords)
    return slat_sparse


def change_format(in_path):
    sptensor = torch.load(in_path)
    fname = in_path.strip(".pt")
    torch.save(sptensor.coords.cpu(), f"{fname}_coords.pt")
    torch.save(sptensor.feats.cpu(), f"{fname}_feats.pt")


def tile_images_2x2(imgs):
    """
    Tile four images (H, W, 3) into a 2×2 NumPy grid.
    Args:
        imgs: list or tuple of 4 images as numpy arrays
    Returns:
        Tiled image as a single numpy array (2H, 2W, 3)
    """
    assert len(imgs) == 4, "Need exactly 4 images"
    h, w, c = imgs[0].shape
    for i in imgs:
        assert i.shape == (h, w, c), "All images must have same shape"

    top = np.concatenate((imgs[0], imgs[1]), axis=1)  # horizontal
    bottom = np.concatenate((imgs[2], imgs[3]), axis=1)
    grid = np.concatenate((top, bottom), axis=0)  # vertical
    return grid


def visualize_obj(gs_decoder, dir, name, save_path, eps):
    suffixes = ["_ori.pt", "_pred.pt", "_gt.pt"]
    labels = ["Input", "Pred", "GT"]
    outs = []
    for suffix in suffixes:
        slat = load_sparse_tensor(f"{dir}/{name}{suffix}", eps=eps).float().cuda()
        # sparsify
        outputs = gs_decoder(slat)#pipeline.decode_slat(slat, ["gaussian"])
        views = render_utils.render_canonical_views(outputs[0])['color']
        viewgrid = tile_images_2x2(views)
        outs.append(viewgrid)
    # concatenate everything in outs
    imgout = np.concatenate(outs, axis=1)

    img = Image.fromarray(imgout)
    draw = ImageDraw.Draw(img)

    # Optional: use a nicer font if available
    font = ImageFont.truetype("DejaVuSans.ttf", size=50)

    # Get dimensions
    img_width, img_height = img.size
    section_width = img_width // 3

    # Draw labels above each third
    label_y = 60
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = section_width * i + (section_width - text_width) // 2
        draw.text((x, label_y), label, fill=(0, 0, 0), font=font)

    # Draw main title centered at top
    title = f"{name}"
    bbox = draw.textbbox((0, 0), label, font=font)
    title_width, _ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((img_width - title_width) // 2 -500, label_y -50), title, fill=(0, 0, 0), font=font)
    img.save(f"{save_path}/{name}.png")
    return

if __name__ == "__main__":
    gs_decoder_save_path = "/data/ziqi/checkpoints/trellis/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
    dir = "/data/ziqi/data/fractaledit/2kbs32w2contpredguidingblr0.002/mariter4,4,4-2k/"
    save_path = f"{dir}/visualize"
    epoch_number = "epoch719"

    os.makedirs(save_path, exist_ok=True)
    gs_decoder = load_gs_decoder(gs_decoder_save_path)
    all_names = list(set(["_".join(name.split("_")[:-1]) for name in os.listdir(dir)]))
    names = [name for name in all_names if epoch_number in name]
    for name in names:
        print(name)
        visualize_obj(gs_decoder, dir, name, save_path, eps=10)

    

