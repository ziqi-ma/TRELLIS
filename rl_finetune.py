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

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
name = "block_edit"
image = Image.open(f"assets/example_image/yellow_block.jpg")

# Run the pipeline
lr = 8e-5
n_epochs = 10
n_inner_epochs=60

#torch.autograd.set_detect_anomaly(True)
outputs = pipeline.train_slat_with_reward(
        n_epochs=n_epochs,
        n_inner_train_epochs=n_inner_epochs,
        lr=lr,
        edit_prompt="the image of a purple cube",
        image=image,
        seed=123
        )
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(f"out/ddpo/{name}_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(f"out/ddpo/{name}_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(f"out/ddpo/{name}_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
'''
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(f"out/{name}.glb")
'''
# Save Gaussians as PLY files
#outputs['gaussian'][0].save_ply("sample.ply")
