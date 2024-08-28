import os
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

# 设置目录路径
base_dir = "/root/autodl-tmp/SAM/earring_pairs"
mask_dir = "/root/autodl-tmp/SAM/mask"
noshadow_dir = "/root/autodl-tmp/SAM/noshadow"

# 创建输出目录（如果不存在）
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(noshadow_dir, exist_ok=True)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="/root/autodl-tmp/SAM/checkpoint/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


# 遍历所有pair文件夹
for pair_folder in sorted(os.listdir(base_dir)):
    pair_path = os.path.join(base_dir, pair_folder)
    
    if not os.path.isdir(pair_path):
        continue

    model_image_path = os.path.join(pair_path, "jewellery.jpg")

    if not os.path.exists(model_image_path):
        print(f"model.jpg 不存在于 {pair_folder}")
        continue

    # 读取并处理图像
    image = Image.open(model_image_path).convert("RGB")
    image_np = np.array(image)

    # 生成蒙版
    try:
        masks = mask_generator.generate(image_np)
        
    except torch.cuda.OutOfMemoryError:
        print(f"显存不足，跳过: {pair_folder}")
        torch.cuda.empty_cache()
        continue

    # 获取第一个蒙版（假设耳环是唯一目标）
    if len(masks) > 0:
        mask = masks[0]['segmentation']
        # 假设首饰是最重要的对象，通常面积较小且亮度较高，可以通过某些特征来选择最合适的蒙版
        # 以下是一个简化的选择逻辑（可以根据实际情况定制）
        # 这里只是简单地选择面积较小的蒙版
        # mask = sorted(masks, key=lambda x: np.sum(x['segmentation']))[1]['segmentation']
    else:
        print(f"无法生成蒙版 {pair_folder}")
        continue

    # 生成保存的文件名
    file_number = pair_folder.split('_')[-1]
    mask_filename = f"model_{file_number}.jpg"
    noshadow_filename = f"model_{file_number}.jpg"

    # 保存蒙版图像
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save(os.path.join(mask_dir, mask_filename))

    # 保存原始图像
    image.save(os.path.join(noshadow_dir, noshadow_filename))

    # 每次处理完一个图像后清空显存
    torch.cuda.empty_cache()

    print(f"处理完成: {pair_folder}")

print("所有处理完成！")

