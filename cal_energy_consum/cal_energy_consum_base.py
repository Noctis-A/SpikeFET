import pickle
import os
class AveragingDict:
    def __init__(self):
        self.sum_dict = {}
        self.count_dict = {}

    def add_dict(self, new_dict):
        for key, value in new_dict.items():
            # 更新总和
            if key in self.sum_dict:
                self.sum_dict[key] += value
                self.count_dict[key] += 1
            else:
                self.sum_dict[key] = value
                self.count_dict[key] = 1

    def get_averages(self):
        avg_dict = {}
        for key in self.sum_dict:
            avg_dict[key] = self.sum_dict[key] / self.count_dict[key]
        return avg_dict
averager = AveragingDict()
seq_path = './fr_dict'
seq_files = os.listdir(seq_path)
for data_file in seq_files:
    with open(os.path.join(seq_path, data_file), "rb") as f:
        loaded_dict = pickle.load(f)
    averager.add_dict(loaded_dict)
fr_dict = averager.get_averages()
fr_dict['backbone.downsample1_1_images.encode_spike'] = 1
fr_dict['backbone.downsample1_1_events.encode_spike'] = 1
fr_dict['box_head_image.conv5_ctr.0'] = 1
fr_dict['box_head_image.conv5_offset.0'] = 1
fr_dict['box_head_image.conv5_size.0'] = 1
fr_dict['box_head_event.conv5_ctr.0'] = 1
fr_dict['box_head_event.conv5_offset.0'] = 1
fr_dict['box_head_event.conv5_size.0'] = 1
# q_k_v的发放率不用，直接用qkv相乘的矩阵代替
FLOPs = {
# downsample1_1_images
'backbone.downsample1_1_images.encode_spike': 231211008,
# ConvBlock1_1_images
'backbone.ConvBlock1_1_images.0.Conv.spike1': 201326592,
'backbone.ConvBlock1_1_images.0.Conv.spike2': 154140672,
'backbone.ConvBlock1_1_images.0.Conv.spike3': 201326592,
'backbone.ConvBlock1_1_images.0.spike1': 3623878656,
'backbone.ConvBlock1_1_images.0.spike2': 3623878656,
# downsample1_2_images
'backbone.downsample1_2_images.encode_spike': 452984832,
# ConvBlock1_2_images
'backbone.ConvBlock1_2_images.0.Conv.spike1': 201326592,
'backbone.ConvBlock1_2_images.0.Conv.spike2': 77070336,
'backbone.ConvBlock1_2_images.0.Conv.spike3': 201326592,
'backbone.ConvBlock1_2_images.0.spike1': 3623878656,
'backbone.ConvBlock1_2_images.0.spike2': 3623878656,
# downsample2_images
'backbone.downsample2_images.encode_spike': 452984832,
# ConvBlock2_1_images
'backbone.ConvBlock2_1_images.0.Conv.spike1': 201326592,
'backbone.ConvBlock2_1_images.0.Conv.spike2': 38535168,
'backbone.ConvBlock2_1_images.0.Conv.spike3': 201326592,
'backbone.ConvBlock2_1_images.0.spike1': 3623878656,
'backbone.ConvBlock2_1_images.0.spike2': 3623878656,
# ConvBlock2_2_images
'backbone.ConvBlock2_2_images.0.Conv.spike1': 201326592,
'backbone.ConvBlock2_2_images.0.Conv.spike2': 38535168,
'backbone.ConvBlock2_2_images.0.Conv.spike3': 201326592,
'backbone.ConvBlock2_2_images.0.spike1': 3623878656,
'backbone.ConvBlock2_2_images.0.spike2': 3623878656,
# downsample3_images
'backbone.downsample3_images.encode_spike': 452984832,
# downsample1_1_events
'backbone.downsample1_1_events.encode_spike': 231211008,
# ConvBlock1_1_events
'backbone.ConvBlock1_1_events.0.Conv.spike1': 201326592,
'backbone.ConvBlock1_1_events.0.Conv.spike2': 154140672,
'backbone.ConvBlock1_1_events.0.Conv.spike3': 201326592,
'backbone.ConvBlock1_1_events.0.spike1': 3623878656,
'backbone.ConvBlock1_1_events.0.spike2': 3623878656,
# downsample1_2_events
'backbone.downsample1_2_events.encode_spike': 452984832,
# ConvBlock1_2_events
'backbone.ConvBlock1_2_events.0.Conv.spike1': 201326592,
'backbone.ConvBlock1_2_events.0.Conv.spike2': 77070336,
'backbone.ConvBlock1_2_events.0.Conv.spike3': 201326592,
'backbone.ConvBlock1_2_events.0.spike1': 3623878656,
'backbone.ConvBlock1_2_events.0.spike2': 3623878656,
# downsample2_events
'backbone.downsample2_events.encode_spike': 452984832,
# ConvBlock2_1_events
'backbone.ConvBlock2_1_events.0.Conv.spike1': 201326592,
'backbone.ConvBlock2_1_events.0.Conv.spike2': 38535168,
'backbone.ConvBlock2_1_events.0.Conv.spike3': 201326592,
'backbone.ConvBlock2_1_events.0.spike1': 3623878656,
'backbone.ConvBlock2_1_events.0.spike2': 3623878656,
# ConvBlock2_2_events
'backbone.ConvBlock2_2_events.0.Conv.spike1': 201326592,
'backbone.ConvBlock2_2_events.0.Conv.spike2': 38535168,
'backbone.ConvBlock2_2_events.0.Conv.spike3': 201326592,
'backbone.ConvBlock2_2_events.0.spike1': 3623878656,
'backbone.ConvBlock2_2_events.0.spike2': 3623878656,
# downsample3_events
'backbone.downsample3_events.encode_spike': 452984832,
# block3_0
'backbone.block3.0.conv.spike1': 402653184,
'backbone.block3.0.conv.spike2': 7077888,
'backbone.block3.0.conv.spike3': 402653184,
'backbone.block3.0.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.0.attn.attns.0.qkv_spike': 1,
'backbone.block3.0.attn.attns.1.qkv_spike': 1,
'backbone.block3.0.attn.attn_spike': 805306368,
'backbone.block3.0.mlp.fc1_spike': 805306368,
'backbone.block3.0.mlp.fc2_spike': 805306368,
# block3_1
'backbone.block3.1.conv.spike1': 402653184,
'backbone.block3.1.conv.spike2': 7077888,
'backbone.block3.1.conv.spike3': 402653184,
'backbone.block3.1.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.1.attn.attns.0.qkv_spike': 1,
'backbone.block3.1.attn.attns.1.qkv_spike': 1,
'backbone.block3.1.attn.attn_spike': 805306368,
'backbone.block3.1.mlp.fc1_spike': 805306368,
'backbone.block3.1.mlp.fc2_spike': 805306368,
# block3_2
'backbone.block3.2.conv.spike1': 402653184,
'backbone.block3.2.conv.spike2': 7077888,
'backbone.block3.2.conv.spike3': 402653184,
'backbone.block3.2.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.2.attn.attns.0.qkv_spike': 1,
'backbone.block3.2.attn.attns.1.qkv_spike': 1,
'backbone.block3.2.attn.attn_spike': 805306368,
'backbone.block3.2.mlp.fc1_spike': 805306368,
'backbone.block3.2.mlp.fc2_spike': 805306368,
# block3_3
'backbone.block3.3.conv.spike1': 402653184,
'backbone.block3.3.conv.spike2': 7077888,
'backbone.block3.3.conv.spike3': 402653184,
'backbone.block3.3.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.3.attn.attns.0.qkv_spike': 1,
'backbone.block3.3.attn.attns.1.qkv_spike': 1,
'backbone.block3.3.attn.attn_spike': 805306368,
'backbone.block3.3.mlp.fc1_spike': 805306368,
'backbone.block3.3.mlp.fc2_spike': 805306368,
# block3_4
'backbone.block3.4.conv.spike1': 402653184,
'backbone.block3.4.conv.spike2': 7077888,
'backbone.block3.4.conv.spike3': 402653184,
'backbone.block3.4.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.4.attn.attns.0.qkv_spike': 1,
'backbone.block3.4.attn.attns.1.qkv_spike': 1,
'backbone.block3.4.attn.attn_spike': 805306368,
'backbone.block3.4.mlp.fc1_spike': 805306368,
'backbone.block3.4.mlp.fc2_spike': 805306368,
# block3_5
'backbone.block3.5.conv.spike1': 402653184,
'backbone.block3.5.conv.spike2': 7077888,
'backbone.block3.5.conv.spike3': 402653184,
'backbone.block3.5.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.5.attn.attns.0.qkv_spike': 1,
'backbone.block3.5.attn.attns.1.qkv_spike': 1,
'backbone.block3.5.attn.attn_spike': 805306368,
'backbone.block3.5.mlp.fc1_spike': 805306368,
'backbone.block3.5.mlp.fc2_spike': 805306368,
# block3_6
'backbone.block3.6.conv.spike1': 402653184,
'backbone.block3.6.conv.spike2': 7077888,
'backbone.block3.6.conv.spike3': 402653184,
'backbone.block3.6.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.6.attn.attns.0.qkv_spike': 1,
'backbone.block3.6.attn.attns.1.qkv_spike': 1,
'backbone.block3.6.attn.attn_spike': 805306368,
'backbone.block3.6.mlp.fc1_spike': 805306368,
'backbone.block3.6.mlp.fc2_spike': 805306368,
# block3_7
'backbone.block3.7.conv.spike1': 402653184,
'backbone.block3.7.conv.spike2': 7077888,
'backbone.block3.7.conv.spike3': 402653184,
'backbone.block3.7.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.7.attn.attns.0.qkv_spike': 1,
'backbone.block3.7.attn.attns.1.qkv_spike': 1,
'backbone.block3.7.attn.attn_spike': 805306368,
'backbone.block3.7.mlp.fc1_spike': 805306368,
'backbone.block3.7.mlp.fc2_spike': 805306368,
# block3_8
'backbone.block3.8.conv.spike1': 402653184,
'backbone.block3.8.conv.spike2': 7077888,
'backbone.block3.8.conv.spike3': 402653184,
'backbone.block3.8.attn.head_spike': 201326592*3,# head_spike
'backbone.block3.8.attn.attns.0.qkv_spike': 1,
'backbone.block3.8.attn.attns.1.qkv_spike': 1,
'backbone.block3.8.attn.attn_spike': 805306368,
'backbone.block3.8.mlp.fc1_spike': 805306368,
'backbone.block3.8.mlp.fc2_spike': 805306368,
# block4_0
'backbone.block4.0.conv.spike1': 402653184,
'backbone.block4.0.conv.spike2': 7077888,
'backbone.block4.0.conv.spike3': 402653184,
'backbone.block4.0.attn.head_spike': 201326592*3,# head_spike
'backbone.block4.0.attn.attns.0.qkv_spike': 1,
'backbone.block4.0.attn.attn_spike': 805306368,
'backbone.block4.0.mlp.fc1_spike': 805306368,
'backbone.block4.0.mlp.fc2_spike': 805306368,
# block4_1
'backbone.block4.1.conv.spike1': 402653184,
'backbone.block4.1.conv.spike2': 7077888,
'backbone.block4.1.conv.spike3': 402653184,
'backbone.block4.1.attn.head_spike': 201326592*3,# head_spike
'backbone.block4.1.attn.attns.0.qkv_spike': 1,
'backbone.block4.1.attn.attn_spike': 805306368,
'backbone.block4.1.mlp.fc1_spike': 805306368,
'backbone.block4.1.mlp.fc2_spike': 805306368,
# block4_2
'backbone.block4.2.conv.spike1': 402653184,
'backbone.block4.2.conv.spike2': 7077888,
'backbone.block4.2.conv.spike3': 402653184,
'backbone.block4.2.attn.head_spike': 201326592*3,# head_spike
'backbone.block4.2.attn.attns.0.qkv_spike': 1,
'backbone.block4.2.attn.attn_spike': 805306368,
'backbone.block4.2.mlp.fc1_spike': 805306368,
'backbone.block4.2.mlp.fc2_spike': 805306368,
# conv1_ctr
'box_head_image.conv1_ctr.0': 212336640,
# conv2_ctr
'box_head_image.conv2_ctr.0': 75497472,
# conv3_ctr
'box_head_image.conv3_ctr.0': 18874368,
# conv4_ctr
'box_head_image.conv4_ctr.0': 4718592,
# conv5_ctr
'box_head_image.conv5_ctr.0': 8192,
# conv1_offset
'box_head_image.conv1_offset.0': 212336640,
# conv2_offset
'box_head_image.conv2_offset.0': 75497472,
# conv3_offset
'box_head_image.conv3_offset.0': 18874368,
# conv4_offset
'box_head_image.conv4_offset.0': 4718592,
# conv5_offset
'box_head_image.conv5_offset.0': 16384,
# conv1_size
'box_head_image.conv1_size.0': 212336640,
# conv2_size
'box_head_image.conv2_size.0': 75497472,
# conv3_size
'box_head_image.conv3_size.0': 18874368,
# conv4_size
'box_head_image.conv4_size.0': 4718592,
# conv5_size
'box_head_image.conv5_size.0': 16384,

# conv1_ctr
'box_head_event.conv1_ctr.0': 212336640,
# conv2_ctr
'box_head_event.conv2_ctr.0': 75497472,
# conv3_ctr
'box_head_event.conv3_ctr.0': 18874368,
# conv4_ctr
'box_head_event.conv4_ctr.0': 4718592,
# conv5_ctr
'box_head_event.conv5_ctr.0': 8192,
# conv1_offset
'box_head_event.conv1_offset.0': 212336640,
# conv2_offset
'box_head_event.conv2_offset.0': 75497472,
# conv3_offset
'box_head_event.conv3_offset.0': 18874368,
# conv4_offset
'box_head_event.conv4_offset.0': 4718592,
# conv5_offset
'box_head_event.conv5_offset.0': 16384,
# conv1_size
'box_head_event.conv1_size.0': 212336640,
# conv2_size
'box_head_event.conv2_size.0': 75497472,
# conv3_size
'box_head_event.conv3_size.0': 18874368,
# conv4_size
'box_head_event.conv4_size.0': 4718592,
# conv5_size
'box_head_event.conv5_size.0': 16384
}
list = []
for key in FLOPs:
    list.append(fr_dict[key] * FLOPs[key])
print(list)
print(sum(list)/1e9*0.9*8)