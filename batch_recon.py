import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
#os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import os

# pipeline currently doesn't support batched, but if we scale this up we need to
'''
def run_obj_edits_batched(obj_name, pipeline):
    imgs = os.listdir(f"/data/ziqi/data/3dedit_engine/edited2d/{obj_name}")
    img_list = []
    for img in imgs:
        image = Image.open(f"/data/ziqi/data/3dedit_engine/edited2d/{obj_name}/{img}")
        img_list.append(image)

    # Run the pipeline
    outputs = pipeline.run(
        img_list,
        seed=1,
        formats=['gaussian']
    )

    # Render the outputs
    for i in range(len(outputs['gaussian'])):
        img_name = imgs[i]
        video = render_utils.render_video(outputs['gaussian'][i])['color']
        image_name_stripped = img_name.replace(".png","").replace(".jpg","").replace(".JPEG","").replace(".jpeg","").replace("'","").replace('"',"")
        imageio.mimsave(f"/data/ziqi/data/3dedit_engine/trellisoutput/{obj_name}/{image_name_stripped}.mp4", video, fps=30)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][i],
            outputs['mesh'][i],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(f"/data/ziqi/data/3dedit_engine/trellisoutput/{obj_name}/{image_name_stripped}.glb")
    return
'''

def run_obj_edits(root_path, slat_save_path, render_save_path, uid, pipeline):
    imgs = os.listdir(f"{root_path}/{uid}")
    os.makedirs(f"{slat_save_path}/{uid}", exist_ok=True)
    os.makedirs(f"{render_save_path}/{uid}", exist_ok=True)

    for img in imgs:
        image = Image.open(f"{root_path}/{uid}/{img}")
        image_name_stripped = img.replace(".png","").replace(".jpg","").replace(".JPEG","").replace(".jpeg","")

        # Run the pipeline
        outputs = pipeline.run(
            image,
            seed=1,
            formats=['gaussian'],
            slat_save_path=f"{slat_save_path}/{uid}/{image_name_stripped}.pt"
        )

        # Render the outputs
        view = render_utils.render_one_view(outputs['gaussian'][0])['color'][0]
        
        im = Image.fromarray(view)
        im.save(f"{render_save_path}/{uid}/{image_name_stripped}.jpg")
  
        '''
        # video and glb for visualization, not to do at scale
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(f"{render_save_path}/{uid}/{image_name_stripped}.mp4", video, fps=30)
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(f"/data/ziqi/data/3dedit_engine/trellisoutput/{obj_name}/{image_name_stripped}.glb")
        '''
    return


def run_obj_original(category, pipeline):
    imgs = os.listdir(f"/data/ziqi/data/3dedit_engine/input/{category}")
    for img in imgs:
        image = Image.open(f"/data/ziqi/data/3dedit_engine/input/{category}/{img}")
        uid = img.replace(".png","").replace(".jpg","").replace(".JPEG","").replace(".jpeg","")
        os.makedirs(f"{slat_save_path}/{uid}", exist_ok=True)
        os.makedirs(f"{render_save_path}/{uid}", exist_ok=True)
        # Run the pipeline
        outputs = pipeline.run(
            image,
            seed=1,
            formats=['gaussian'],
            slat_save_path=f"{slat_save_path}/{uid}/original.pt"
        )

        # Render the outputs
        view = render_utils.render_one_view(outputs['gaussian'][0])['color'][0]
        
        im = Image.fromarray(view)
        im.save(f"{render_save_path}/{uid}/original.jpg")
        
    return


if __name__ == '__main__':
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    root_path = "/data/ziqi/data/3dedit_engine/edited2d/car"
    slat_save_path = "/data/ziqi/data/3dedit_engine/slats/car"
    render_save_path = "/data/ziqi/data/3dedit_engine/recon_rendered/car"

    uids = os.listdir("/data/ziqi/data/3dedit_engine/edited2d/car")
    print(len(uids))

    i = 0

    #for uid in uids:
        #run_obj_edits(root_path, slat_save_path, render_save_path, uid, pipeline)
        #i += 1
        #print(f"{i} out of {len(uids)} done")


    #obj_list = ["6","9","11","12", "ballpoint", "crane", "goldfish", "grand_piano", "lampshade", "pencil_box", "traffic_light", "Yorkshire_terrier"]

    #for obj_name in obj_list:
        #run_obj_edits(obj_name, pipeline)

    run_obj_original("car", pipeline)
