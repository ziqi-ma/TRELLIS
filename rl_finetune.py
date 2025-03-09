import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch
# Load a pipeline from a model folder or a Hugging Face model hub.
import time

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
name = "block_edit"
image = Image.open(f"assets/example_image/yellow_block.jpg")

# Run the pipeline
lr = 3e-4
n_epochs = 50
n_inner_epochs=1

stime = time.time()
#torch.autograd.set_detect_anomaly(True)
outputs = pipeline.train_slat_with_reward(
        n_epochs=n_epochs,
        n_inner_train_epochs=n_inner_epochs,
        lr=lr,
        edit_prompt="purple",
        image=image,
        steps=100,
        seed=123,
        inner_batch_size = 50
        )

etime = time.time()
print(etime-stime)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

