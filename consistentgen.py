import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def exp_pair(names, start_step_ss, start_layer_ss, stop_layer_ss, start_step_l, start_layer_l, stop_layer_l, seed):
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    images = [Image.open(f"../Hunyuan3D-2/input/{name}.png") for name in names]

    # Run the pipeline
    outputs = pipeline.run_consistent_all(
        images,
        seed=seed,
        starting_step_ss=start_step_ss,
        starting_layer_ss=start_layer_ss,
        stop_layer_ss=stop_layer_ss,
        starting_step_l=start_step_l,
        starting_layer_l=start_layer_l,
        stop_layer_l=stop_layer_l
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    
    save_prefix = f"/data/ziqi/data/trellismasa/paircrown2back_all{seed}"
    os.makedirs(f"{save_prefix}/sstep{start_step_l}/sl{start_layer_l}", exist_ok=True)
    for i in range(2):
        name = names[i]
        video = render_utils.render_video(outputs['gaussian'][i])['color']
        imageio.mimsave(f"{save_prefix}/sstep{start_step_l}/sl{start_layer_l}/{name}_gs{stop_layer_l}.mp4", video, fps=30)
        #video = render_utils.render_video(outputs['radiance_field'][0])['color']
        #imageio.mimsave(f"out/{name}_rf.mp4", video, fps=30)
        #video = render_utils.render_video(outputs['mesh'][i])['normal']
        #imageio.mimsave(f"{save_prefix}/sstep{start_step_l}/sl{start_layer_l}/{name}_mesh{stop_layer_l}.mp4", video, fps=30)
    
    '''
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export("sample.glb")

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")
    '''

if __name__=="__main__":
    names = ["catbackcrown", "catback"]
    ssteps = [0]#[0,10,20]
    slayers = [0]
    stoplayers = [2,4,6,8]#[0,2,10,24,32,48]#[0,2,4,6,8,10,12,16,24,32]
    seed = 123
    for sstep in ssteps:
        for slayer in slayers:
            for stoplayer in stoplayers:
                if stoplayer <= slayer:
                    continue
                exp_pair(names, 0, 0, 4, sstep, slayer, stoplayer, seed)