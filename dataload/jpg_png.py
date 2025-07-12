import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # 构建完整的文件路径
            jpg_file_path = os.path.join(input_folder, filename)
            png_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

            # 打开JPEG文件并转换为PNG
            with Image.open(jpg_file_path) as img:
                img.convert('RGBA').save(png_file_path, 'PNG')
                print(f"Converted {jpg_file_path} to {png_file_path}")

            # 删除原始JPEG文件
            os.remove(jpg_file_path)
            print(f"Deleted {jpg_file_path}")

# 使用示例
input_folder = '/data_share/ymr/pycharm/romote_data/DIOR'  # 替换为你的JPEG文件夹路径
output_folder = '/data_share/ymr/pycharm/romote_data/DIOR'  # 替换为你想要保存PNG文件的文件夹路径
convert_jpg_to_png(input_folder, output_folder)
