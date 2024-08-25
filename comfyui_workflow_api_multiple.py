import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    VAEEncodeForInpaint,
    VAEDecode,
    KSampler,
    CLIPTextEncode,
    CLIPVisionLoader,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    LoadImage,
)


def main():
    import_custom_nodes()

    # 设置输入和输出路径
    dataset_dir = 'earring_pairs'  # 数据集路径
    output_dir = 'output'  # 输出路径

    # 如果输出路径不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有pair文件夹，并按照名称排序
    pair_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))],
                          key=lambda x: int(x.split('_')[-1]))
    
    # 遍历所有的图片对
    for pair_folder in pair_folders:
        pair_path = os.path.join(dataset_dir, pair_folder)
        
        # 确保当前项是一个目录
        if not os.path.isdir(pair_path):
            continue

        # 获取model.jpg和jewellery.jpg的路径
        model_image_path = os.path.join(pair_path, 'model.jpg')
        jewellery_image_path = os.path.join(pair_path, 'jewellery.jpg')

        if not os.path.exists(model_image_path) or not os.path.exists(jewellery_image_path):
            print(f"Skipping {pair_folder}: missing images.")
            continue

        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_1 = loadimage.load_image(image=model_image_path)

            sammodelloader_segment_anything = NODE_CLASS_MAPPINGS[
                "SAMModelLoader (segment anything)"
            ]()
            sammodelloader_segment_anything_3 = sammodelloader_segment_anything.main(
                model_name="sam_vit_h (2.56GB)"
            )

            groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS[
                "GroundingDinoModelLoader (segment anything)"
            ]()
            groundingdinomodelloader_segment_anything_4 = (
                groundingdinomodelloader_segment_anything.main(
                    model_name="GroundingDINO_SwinB (938MB)"
                )
            )

            loadimage_8 = loadimage.load_image(image=jewellery_image_path)

            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_11 = checkpointloadersimple.load_checkpoint(
                ckpt_name="albedobaseXL_v13.safetensors"
            )

            clipvisionloader = CLIPVisionLoader()
            clipvisionloader_12 = clipvisionloader.load_clip(
            clip_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            )

            cliptextencode = CLIPTextEncode()
            cliptextencode_14 = cliptextencode.encode(
            text="earrings", clip=get_value_at_index(checkpointloadersimple_11, 1)
            )

            cliptextencode_15 = cliptextencode.encode(
            text="", clip=get_value_at_index(checkpointloadersimple_11, 1)
            )

            groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoSAMSegment (segment anything)"
            ]()
            groundingdinosamsegment_segment_anything_2 = (
            groundingdinosamsegment_segment_anything.main(
                prompt="earrings",
                threshold=0.3,
                sam_model=get_value_at_index(sammodelloader_segment_anything_3, 0),
                grounding_dino_model=get_value_at_index(
                    groundingdinomodelloader_segment_anything_4, 0
                ),
                image=get_value_at_index(loadimage_1, 0),
            )
            )

            feathermask = NODE_CLASS_MAPPINGS["FeatherMask"]()
            feathermask_5 = feathermask.feather(
            left=2,
            top=2,
            right=2,
            bottom=2,
            mask=get_value_at_index(groundingdinosamsegment_segment_anything_2, 1),
            )

            vaeencodeforinpaint = VAEEncodeForInpaint()
            vaeencodeforinpaint_16 = vaeencodeforinpaint.encode(
            grow_mask_by=6,
            pixels=get_value_at_index(loadimage_1, 0),
            vae=get_value_at_index(checkpointloadersimple_11, 2),
            mask=get_value_at_index(feathermask_5, 0),
            )

            ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
            ]()
            ultralyticsdetectorprovider_22 = ultralyticsdetectorprovider.doit(
            model_name="bbox/deepfashion2_yolov8s-seg.pt"
            )

            samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
            samloader_23 = samloader.load_model(
            model_name="sam_vit_h_4b8939.pth", device_mode="AUTO"
            )

            masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
            ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
            ksampler = KSampler()
            vaedecode = VAEDecode()
            facedetailer = NODE_CLASS_MAPPINGS["FaceDetailer"]()

            for q in range(1):
                masktoimage_6 = masktoimage.mask_to_image(
                    mask=get_value_at_index(feathermask_5, 0)
                )

                ipadapterunifiedloader_10 = ipadapterunifiedloader.load_models(
                    preset="PLUS (high strength)",
                    model=get_value_at_index(checkpointloadersimple_11, 0),
                )

                ipadapteradvanced_9 = ipadapteradvanced.apply_ipadapter(
                    weight=1,
                    weight_type="style transfer",
                    combine_embeds="concat",
                    start_at=0,
                    end_at=1,
                    embeds_scaling="V only",
                    model=get_value_at_index(ipadapterunifiedloader_10, 0),
                    ipadapter=get_value_at_index(ipadapterunifiedloader_10, 1),
                    image=get_value_at_index(loadimage_8, 0),
                    attn_mask=get_value_at_index(feathermask_5, 0),
                    clip_vision=get_value_at_index(clipvisionloader_12, 0),
                )

                ksampler_13 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=30,
                    cfg=8,
                    sampler_name="dpmpp_2m",
                    scheduler="karras",
                    denoise=1,
                    model=get_value_at_index(ipadapteradvanced_9, 0),
                    positive=get_value_at_index(cliptextencode_14, 0),
                    negative=get_value_at_index(cliptextencode_15, 0),
                    latent_image=get_value_at_index(vaeencodeforinpaint_16, 0),
                )

                vaedecode_17 = vaedecode.decode(
                    samples=get_value_at_index(vaeencodeforinpaint_16, 0),
                    vae=get_value_at_index(checkpointloadersimple_11, 2),
                )

                vaedecode_19 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_13, 0),
                    vae=get_value_at_index(checkpointloadersimple_11, 2),
                )

                facedetailer_21 = facedetailer.doit(
                    guide_size=384,
                    guide_size_for=True,
                    max_size=1024,
                    seed=random.randint(1, 2**64),
                    steps=20,
                    cfg=8,
                    sampler_name="euler",
                    scheduler="normal",
                    denoise=0.3,
                    feather=5,
                    noise_mask=True,
                    force_inpaint=True,
                    bbox_threshold=0.5,
                    bbox_dilation=10,
                    bbox_crop_factor=3,
                    sam_detection_hint="center-1",
                    sam_dilation=0,
                    sam_threshold=0.93,
                    sam_bbox_expansion=0,
                    sam_mask_hint_threshold=0.7,
                    sam_mask_hint_use_negative="False",
                    drop_size=10,
                    wildcard="",
                    cycle=1,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    image=get_value_at_index(vaedecode_19, 0),
                    model=get_value_at_index(checkpointloadersimple_11, 0),
                    clip=get_value_at_index(checkpointloadersimple_11, 1),
                    vae=get_value_at_index(checkpointloadersimple_11, 2),
                    positive=get_value_at_index(cliptextencode_14, 0),
                    negative=get_value_at_index(cliptextencode_15, 0),
                    bbox_detector=get_value_at_index(ultralyticsdetectorprovider_22, 0),
                    sam_model_opt=get_value_at_index(samloader_23, 0),
                )

                masktoimage_25 = masktoimage.mask_to_image(
                    mask=get_value_at_index(facedetailer_21, 3)
                )

                from torchvision.transforms import ToPILImage

                print(facedetailer_21[0].shape)

                image_tensor = facedetailer_21[0][0]  

                rearranged_tensor = image_tensor.permute(2,0,1)
                try:
                    pil_image = ToPILImage()(rearranged_tensor)
                    pair_number = pair_folder.split('_')[-1]  # 提取pair的数字
                    output_image_path = os.path.join(output_dir, f'pair_{pair_number}.jpg')
                    # output_image_path = os.path.join(output_dir, f'{pair_folder}.jpg')
                    pil_image.save(output_image_path)
                    print(f"Image saved to {output_image_path}")
                except ValueError as e:
                    print(e)


if __name__ == "__main__":
    main()
