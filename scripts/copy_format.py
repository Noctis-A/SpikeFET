from tqdm import tqdm
import os
import shutil

def copy_folder_excluding_aedat4(src, dst, aedat4_file_to_keep):
    # 创建目标文件夹（如果不存在）
    os.makedirs(dst, exist_ok=True)
    data_list = sorted(os.listdir(src))
    for foldName in tqdm(data_list):
        src_list = os.path.join(src, foldName)
        dst_save = os.path.join(dst, foldName)
        os.makedirs(dst_save, exist_ok=True)
        # 复制源文件夹中的所有内容到目标文件夹
        for item in os.listdir(src_list):
            s = os.path.join(src_list, item)
            d = os.path.join(dst_save, item)
            # # 检查是否是要保留的 .aedat4 文件
            # if os.path.isfile(s) and item.split('.')[-1] == aedat4_file_to_keep:
            #     print(f"保留文件: {s}")
            #     continue  # 跳过要保留的文件
            if os.path.isdir(s) and item == "evt":
                # 复制文件或文件夹
                shutil.copytree(s, d, dirs_exist_ok=True)  # 复制子文件夹
                shutil.rmtree(s)  # 复制子文件夹

            # # 复制文件或文件夹
            # if os.path.isdir(s):
            #     shutil.copytree(s, d, dirs_exist_ok=True)  # 复制子文件夹
            # else:
            #     shutil.copy2(s, d)  # 复制文件

# 示例用法
source_folder = '/home/work/xxx/VisEvent/train'  # 源文件夹路径
destination_folder = '/home/work/xxx/VisEvent/train'  # 目标文件夹路径
aedat4_file_to_keep = 'aedat4'  # 要保留的 .txt 文件名

copy_folder_excluding_aedat4(source_folder, destination_folder, aedat4_file_to_keep)
