import os
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageTk

# 初始化Tkinter窗口
root = Tk()
root.title("Drag to Position Jewellery")

# 加载模型图片和耳饰图片
model_folder = "/Users/eureka/VSCodeProjects/ARShadowGAN/data_model"
jewellery_img_path = "/Users/eureka/VSCodeProjects/ARShadowGAN/jewellery.png"
save_folder = "/Users/eureka/VSCodeProjects/ARShadowGAN/processed_images"
mask_folder = "/Users/eureka/VSCodeProjects/ARShadowGAN/masks"

os.makedirs(save_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# 获取所有模型图片路径
model_images = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith(('.png', '.jpg'))]

# 初始化Canvas和图片
canvas = Canvas(root, width=800, height=600)
canvas.pack()

def load_images():
    global model_img, jewellery_img, model_photo, jewellery_photo, current_model_img_path
    if model_images:
        # 清除画布上的所有内容
        canvas.delete("all")

        # 加载当前模型图片
        current_model_img_path = model_images.pop(0)
        model_img = Image.open(current_model_img_path)
        model_img.thumbnail((800, 600))
        model_photo = ImageTk.PhotoImage(model_img)
        
        # 加载耳饰图片
        jewellery_img = Image.open(jewellery_img_path)
        jewellery_photo = ImageTk.PhotoImage(jewellery_img)
        
        # 在Canvas上显示图片
        canvas.create_image(0, 0, image=model_photo, anchor='nw')
        canvas.create_image(0, 0, image=jewellery_photo, anchor='nw', tags="jewellery")

        # 使耳饰图片可拖拽
        canvas.tag_bind("jewellery", "<Button1-Motion>", on_drag)

def on_drag(event):
    x, y = event.x, event.y
    canvas.coords("jewellery", x, y)

def save_image():
    x, y = canvas.coords("jewellery")
    # 创建带耳饰的最终图片
    final_img = model_img.copy()
    final_img.paste(jewellery_img, (int(x), int(y)), jewellery_img)
    final_img_path = os.path.join(save_folder, os.path.basename(current_model_img_path))
    final_img.save(final_img_path)

    # 创建耳饰的蒙版图片
    mask_img = Image.new("L", model_img.size, 0)
    mask_img.paste(jewellery_img.split()[-1], (int(x), int(y)))
    mask_img_path = os.path.join(mask_folder, os.path.basename(current_model_img_path))
    mask_img.save(mask_img_path)

    # 加载下一张图片
    load_images()

# 按钮来保存图片并加载下一张
save_button = Button(root, text="Save and Next", command=save_image)
save_button.pack()

# 加载第一张图片
load_images()

# 运行Tkinter主循环
root.mainloop()