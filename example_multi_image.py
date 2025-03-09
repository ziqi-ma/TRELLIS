import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
import torch

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
images = [
    Image.open(f"/data/ziqi/data/3dwild/pill/pill1.png"),
    Image.open(f"/data/ziqi/data/3dwild/pill/pill2.png"),
    Image.open(f"/data/ziqi/data/3dwild/pill/pill3.png"),
    Image.open(f"/data/ziqi/data/3dwild/pill/pill4.png")
]

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes
extrinsics = [
    torch.tensor(
        [[-0.23558697, -0.11236251,  0.96533597, -0.26432055],
        [ 0.15947382,  0.9753603,   0.15244836, -0.03969496],
        [-0.9586798,   0.18986064, -0.21186325,  0.3159111],
        [ 0.,          0.,          0.,          1.        ]]).cuda(),
    torch.tensor(
        [[ 0.9996055,   0.02296975,  0.01617419, -0.02144218],
        [-0.02801195,  0.85864305,  0.5118083,  -0.13048404],
        [-0.00213175, -0.51205933,  0.8589476,   0.03684792],
        [ 0.,          0.,          0.,          1.        ]]).cuda(),
    torch.tensor(
        [[ 0.99992037, -0.0086343,  -0.00922067,  0.        ],
        [ 0.00854718,  0.999919,   -0.00944689,  0.        ],
        [ 0.00930149,  0.00936733,  0.99991304,  0.        ],
        [ 0.,          0.,          0.,          1.        ]]).cuda(),
    torch.tensor(
        [[ 0.9995066,   0.03123808,  0.00330985, -0.0123569],
        [-0.00517179,  0.0597136,   0.9982023,  -0.26520735],
        [ 0.03098427, -0.9977268,   0.05984569,  0.26004702],
        [ 0.,          0.,          0.,          1.        ]]).cuda(),
    torch.tensor(
        [[-0.29537353,  0.09664372, -0.95048124,  0.22803217],
        [-0.10335968,  0.9857986,   0.13235502, -0.03961044],
        [ 0.9497743,   0.13733561, -0.28118977,  0.33142996],
        [ 0.,          0.,          0.,          1.,        ]]).cuda()
]

fovs = [587.4244, 550.6370, 582.3639, 519.0763, 508.4023]

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("wild/out/pill_nomask.mp4", video, fps=30)
