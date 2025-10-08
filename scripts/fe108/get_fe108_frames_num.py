import os
import numpy as np

data_path = r"/mnt/data/datasets/FE108_dataset/test"
frame_num=0
with open(r"/scripts/fe108/test.txt", 'r') as f:
    video_files = f.readlines()
    video_files = [x.strip() for x in video_files]
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    read_path = os.path.join(data_path, foldName, 'img')
    lenth = len(os.listdir(read_path))
    frame_num += lenth
print(frame_num)

#  train: 138585     test: 59688   total: 198273
