import os
import numpy as np

data_path = r"/home/work/xxx/VisEvent/train"
frame_num=0
with open(r"/home/work/xxx/VisEvent/train/list.txt", 'r') as f:
    video_files = f.readlines()
    video_files = [x.strip() for x in video_files]
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    read_path = os.path.join(data_path, foldName, 'vis_imgs')
    lenth = len(os.listdir(read_path))
    frame_num += lenth
print(frame_num)

#  train: 122852     test: 106836   total: 229688
#  train: 212062     test: 157799   total: 369861