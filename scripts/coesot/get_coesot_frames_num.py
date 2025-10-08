import os
import pdb

data_path = r"/home/work/yjj/COESOT/train"
frame_num=0
video_files = os.listdir(data_path)
for videoID in range(len(video_files)):
    foldName = video_files[videoID]
    if 'txt' in foldName:
        continue
    # print("==>> finished: ", foldName)
    read_path = os.path.join(data_path, foldName, foldName+'_aps')
    lenth = len(os.listdir(read_path))
    frame_num += lenth
print(frame_num)

#  train: 290013     test: 176535   total: 478721
