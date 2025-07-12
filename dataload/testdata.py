import os
from PIL import Image

WHU_RS19 = [
    'Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 'Farmland', 'footballField','Forest',
    'Industrial', 'Meadow', 'Mountain', 'Park', 'Parking', 'Pond', 'Port',
    'railwayStation', 'Residential', 'River', 'Viaduct'
]

# ======== 路径配置 ========
original_HR_folder = '/data_share/ymr/pycharm/romote_data/WHU_RS19/'
save_GT_folder = '/data_share/ymr/pycharm/romote_data/whu_testx4/GT/'
save_LR_folder = '/data_share/ymr/pycharm/romote_data/whu_testx4/LR/'
save_bicubic_folder = '/data_share/ymr/pycharm/romote_data/whu_testx4/Bicubic/'

# ======== 创建所有输出目录 ========
os.makedirs(save_GT_folder, exist_ok=True)
os.makedirs(save_LR_folder, exist_ok=True)
os.makedirs(save_bicubic_folder, exist_ok=True)

# ======== 参数配置 ========
crop_size = 512
up_scale = 4

# ======== 处理每个类别 ========
for class_name in WHU_RS19:
    raw_hr_path = os.path.join(original_HR_folder, class_name)
    
    # ===== 创建子目录 =====
    gt_path = os.path.join(save_GT_folder, class_name)
    lr_path = os.path.join(save_LR_folder, class_name)
    bicubic_path = os.path.join(save_bicubic_folder, class_name)
    
    for p in [gt_path, lr_path, bicubic_path]:
        os.makedirs(p, exist_ok=True)

    # ===== 遍历原始HR图像 =====
    for img_name in os.listdir(raw_hr_path):
        if not img_name.lower().endswith('.jpg'):
            continue
            
        # 生成新文件名（PNG格式）
        base_name = os.path.splitext(img_name)[0]
        new_name = f"{base_name}.png"

        # 加载原始HR图像
        hr_img = Image.open(os.path.join(raw_hr_path, img_name))
        
        # ===== 中心裁剪 =====
        width, height = hr_img.size
        if width < crop_size or height < crop_size:
            print(f"跳过 {img_name} (尺寸不足)")
            continue
            
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        hr_cropped = hr_img.crop((left, top, left+crop_size, top+crop_size))
        
        # ===== 保存裁剪后的GT =====
        hr_cropped.save(os.path.join(gt_path, new_name))
        
        # ===== 生成LR图像 =====
        lr_size = (crop_size // up_scale, crop_size // up_scale)
        lr_img = hr_cropped.resize(lr_size, Image.BICUBIC)
        lr_img.save(os.path.join(lr_path, new_name))
        
        # ===== 生成Bicubic重建图像 =====
        bicubic_img = lr_img.resize((crop_size, crop_size), Image.BICUBIC)
        bicubic_img.save(os.path.join(bicubic_path, new_name))
