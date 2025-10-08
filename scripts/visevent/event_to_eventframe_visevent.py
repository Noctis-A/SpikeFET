import time
from dv import AedatFile
import numpy as np
import cv2
import os
import numba

@numba.jit(nopython=True)
def sigmoid(x):
    return 255 * (1 / (1 + np.exp(-x / 2)))


@numba.jit(nopython=True)
def event_time_image(events, num_bins=3, width=304, height=240):
    # 初始化体素网格（直接三维数组更高效）
    voxel_grid = np.zeros((height, width, num_bins), dtype=np.float32)

    # 计算时间相关参数
    first_stamp = events[0, 0]
    last_stamp = events[-1, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    # 转换坐标和时间到整数索引
    ts = num_bins * (events[:, 0] - first_stamp) / deltaT
    xs = np.clip(events[:, 1].astype(np.int32), 0, width - 1)
    ys = np.clip(events[:, 2].astype(np.int32), 0, height - 1)
    tis = np.clip(ts.astype(np.int32), 0, num_bins - 1)
    pols = events[:, 3].astype(np.float32)
    pols[pols == 0] = -1  # 极性转换为 +1/-1

    # 显式循环替代 np.add.at
    for i in range(len(events)):
        x = xs[i]
        y = ys[i]
        t = tis[i]
        p = pols[i]
        voxel_grid[y, x, t] += p

    # 应用 sigmoid 并转换格式
    voxel_grid = sigmoid(voxel_grid)
    voxel_grid = (voxel_grid).astype(np.uint8)
    return voxel_grid

if __name__ == '__main__':
    height, width = 260, 346
    data_path = r"/mnt/data/datasets/VisEvent_dataset/data/train"
    save_path = r"/home/work/xxx/VisEvent_new/train"
    video_files = os.listdir(data_path)
    dvs_img_interval = 1

    start_time = time.time()
    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        print("==>> foldName: ", foldName)

        if not os.path.exists(os.path.join(save_path, foldName, 'evt')):
            os.makedirs(os.path.join(save_path, foldName, 'evt'))

        evt_save = os.path.join(save_path, foldName, 'evt/')

        aedat4_file = foldName + '.aedat4'
        read_path = os.path.join(data_path, foldName, aedat4_file)

        frame_all = []
        frame_interval_time = []
        with AedatFile(read_path) as f:
            for frame in f['frames']:
                frame_all.append(frame.image)
                frame_interval_time.append([frame.timestamp_start_of_frame,
                                            frame.timestamp_end_of_frame])
        frame_timestamp = frame_interval_time
        frame_num = len(frame_timestamp)

        events = np.hstack([packet for packet in f['events'].numpy()])

        t_all = events['timestamp']
        x_all = events['x']
        y_all = events['y']
        p_all = events['polarity']

        for frame_no in range(0, int(frame_num / dvs_img_interval) - 1):
            start_idx = frame_timestamp[frame_no][0]
            end_idx = frame_timestamp[frame_no][1]
            # end_idx = start_idx + 40000
            mask = (t_all >= start_idx) & (t_all < end_idx)

            t = t_all[mask]
            x = x_all[mask]
            y = y_all[mask]
            p = p_all[mask]
            evs_50 = np.stack((t, x, y, p), axis=1)

            if evs_50.shape[0] == 0:
                img_event_time_image_50 = np.ones((height, width, 3), dtype=np.uint8) * 127
                print('empty event frame ', frame_no)
            else:
                img_event_time_image_50 = event_time_image(evs_50, num_bins=3, width=width, height=height)
                img_event_time_image_50 = cv2.cvtColor(img_event_time_image_50, cv2.COLOR_RGB2BGR)
            file_output_path_event_time_image_50 = os.path.join(evt_save + 'frame{:0>4d}.png'.format(frame_no))
            cv2.imwrite(file_output_path_event_time_image_50, img_event_time_image_50)
    end_time = time.time()
    print("==>> time cost: ", (end_time - start_time) / 60)