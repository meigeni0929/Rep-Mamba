import os
from PIL import Image

# ======== 路径配置 ========
original_HR_folder = '/data_share/ymr/pycharm/romote_data/DIOR/'
save_GT_folder = '/data_share/ymr/pycharm/romote_data/DIORtestx4/GT/'
save_LR_folder = '/data_share/ymr/pycharm/romote_data/DIORtestx4/LR/'
save_bicubic_folder = '/data_share/ymr/pycharm/romote_data/DIORtestx4/Bicubic/'

# ======== 创建所有输出目录 ========
os.makedirs(save_GT_folder, exist_ok=True)
os.makedirs(save_LR_folder, exist_ok=True)
os.makedirs(save_bicubic_folder, exist_ok=True)

# ======== 参数配置 ========
crop_size = 512
up_scale = 4

# ======== 处理所有图像 ========
for img_name in os.listdir(original_HR_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    
    # 获取无扩展文件名
    base_name = os.path.splitext(img_name)[0]
    output_name = f"{base_name}.png"  # 统一改为PNG格式
    
    # 加载原始HR图像
    hr_img = Image.open(os.path.join(original_HR_folder, img_name))
    
    # ===== 中心裁剪 =====
    width, height = hr_img.size
    if width < crop_size or height < crop_size:
        print(f"跳过 {img_name} (尺寸不足)")
        continue
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    hr_cropped = hr_img.crop((left, top, left + crop_size, top + crop_size))
    
    # ===== 保存裁剪后的GT =====
    hr_cropped.save(os.path.join(save_GT_folder, output_name), format='PNG')
    
    # ===== 生成LR图像 =====
    lr_size = (crop_size // up_scale, crop_size // up_scale)
    lr_img = hr_cropped.resize(lr_size, Image.BICUBIC)
    lr_img.save(os.path.join(save_LR_folder, output_name), format='PNG')
    
    # ===== 生成Bicubic重建图像 =====
    bicubic_img = lr_img.resize((crop_size, crop_size), Image.BICUBIC)
    bicubic_img.save(os.path.join(save_bicubic_folder, output_name), format='PNG')

print("处理完成！所有图像已保存为PNG格式")
