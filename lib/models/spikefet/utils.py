import math
import random
import torch
import torch.nn.functional as F
revise_keys = [
    (r'^downsample1_1\.', [r'downsample1_1_images.',r'downsample1_1_events.']),
    (r'^ConvBlock1_1\.', [r'ConvBlock1_1_images.',r'ConvBlock1_1_events.']),
    (r'^downsample1_2\.', [r'downsample1_2_images.',r'downsample1_2_events.']),
    (r'^ConvBlock1_2\.', [r'ConvBlock1_2_images.',r'ConvBlock1_2_events.']),
    (r'^downsample2\.', [r'downsample2_images.',r'downsample2_events.']),
    (r'^ConvBlock2_1\.', [r'ConvBlock2_1_images.',r'ConvBlock2_1_events.']),
    (r'^ConvBlock2_2\.', [r'ConvBlock2_2_images.',r'ConvBlock2_2_events.']),
    (r'^downsample3\.', [r'downsample3_images.',r'downsample3_events.'])
]

revise_keys_finetune = [
    (r'backbone\.downsample1_1\.', [r'backbone.downsample1_1_images.',r'backbone.downsample1_1_events.']),
    (r'backbone\.ConvBlock1_1\.', [r'backbone.ConvBlock1_1_images.',r'backbone.ConvBlock1_1_events.']),
    (r'backbone\.downsample1_2\.', [r'backbone.downsample1_2_images.',r'backbone.downsample1_2_events.']),
    (r'backbone\.ConvBlock1_2\.', [r'backbone.ConvBlock1_2_images.',r'backbone.ConvBlock1_2_events.']),
    (r'backbone\.downsample2\.', [r'backbone.downsample2_images.',r'backbone.downsample2_events.']),
    (r'backbone\.ConvBlock2_1\.', [r'backbone.ConvBlock2_1_images.',r'backbone.ConvBlock2_1_events.']),
    (r'backbone\.ConvBlock2_2\.', [r'backbone.ConvBlock2_2_images.',r'backbone.ConvBlock2_2_events.']),
    (r'backbone\.downsample3\.', [r'backbone.downsample3_images.',r'backbone.downsample3_events.']),
    (r'^box_head\.', [r'box_head_image.',r'box_head_event.'])
]


def combine_tokens(template_tokens, search_tokens, event_z, event_x, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens, event_z, event_x), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature


def recover_tokens(merged_tokens, len_template_token, len_search_token, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    elif mode == 'template_central':
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)
    elif mode == 'partition':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens


def add_modal_embed(x, event_x, modal_embedding, random_id):
    # [B, C, H, W]
    B, C, H, W = x.shape
    H_z, W_z = 8, 8
    H_x, W_x = 16, 16
    if random_id == 0:
        i = torch.zeros(1, H_x * (W_x + W_z), device=x.device).long()
        e = torch.ones(1, H_x * (W_x + W_z), device=x.device).long()
        i = modal_embedding(i)
        e = modal_embedding(e)
        i = i.transpose(1, 2).view(1, C, H_x, W_x + W_z)
        e = e.transpose(1, 2).view(1, C, H_x, W_x + W_z)
        x += i
        event_x += e

    elif random_id == 1: # 上下
        i = torch.zeros(1, (H_x + H_z) * W_x, device=x.device).long()
        e = torch.ones(1, (H_x + H_z) * W_x, device=x.device).long()
        i = modal_embedding(i)
        e = modal_embedding(e)
        i = i.transpose(1, 2).view(1, C, H_x + H_z, W_x)
        e = e.transpose(1, 2).view(1, C, H_x + H_z, W_x)
        x += i
        event_x += e
    return x, event_x


def batch_combine_img(z, x, random_id):
    # [B, C, H, W]
    B_x, C_x, H_x, W_x = x.shape
    B_z, C_z, H_z, W_z = z[0].shape
    
    random_list = []
    if random_id == 0: # 左右
        inps = torch.zeros(B_x, C_x, H_x, W_x + W_z, device=x.device, dtype=x.dtype)
        for i in range(B_x):
            inp = torch.zeros(C_x, H_x, W_x + W_z, device=x.device, dtype=x.dtype)
            random_idx = random.choice(list(range(2)))
            if random_idx == 0: # 左1右2
                inp[:, :, :W_x] = x[i]
                inp[:, :H_z, W_x:] = z[0][i]
                inp[:, H_z:, W_x:] = z[1][i]
            else: # 左2右1
                inp[:, :, W_z:] = x[i]
                inp[:, :H_z, :W_z] = z[0][i]
                inp[:, H_z:, :W_z] = z[1][i]
            inps[i] = inp
            random_list.append(random_idx)

    elif random_id == 1: # 上下
        inps = torch.zeros(B_x, C_x, H_x + H_z, W_x, device=x.device, dtype=x.dtype)
        for i in range(B_x):
            inp = torch.zeros(C_x, H_x + H_z, W_x, device=x.device, dtype=x.dtype)
            random_idx = random.choice(list(range(2, 4)))
            if random_idx == 2:  # 上1下2
                inp[:, :H_x, :] = x[i]
                inp[:, H_x:, :W_z] = z[0][i]
                inp[:, H_x:, W_z:] = z[1][i]
            else:  # 上2下1
                inp[:, H_z:, :] = x[i]
                inp[:, :H_z, :W_z] = z[0][i]
                inp[:, :H_z, W_z:] = z[1][i]
            inps[i] = inp
            random_list.append(random_idx)
    return inps, random_list


def batch_add_pos_embed(inps, random_list, pos_embed_x, pos_embed_z):
    # [B, C, H, W]
    H_z, W_z = 8, 8
    H_x, W_x = 16, 16
    for i in range(len(random_list)):
        random_idx = random_list[i]
        if random_idx == 0:
            inps[i, :, :, :W_x] += pos_embed_x.squeeze(0)
            inps[i, :, :H_z, W_x:] += pos_embed_z.squeeze(0)
            inps[i, :, H_z:, W_x:] += pos_embed_z.squeeze(0)
        elif random_idx == 1:
            inps[i, :, :, W_z:] += pos_embed_x.squeeze(0)
            inps[i, :, :H_z, :W_z] += pos_embed_z.squeeze(0)
            inps[i, :, H_z:, :W_z] += pos_embed_z.squeeze(0)
        elif random_idx == 2:
            inps[i, :, :H_x, :] += pos_embed_x.squeeze(0)
            inps[i, :, H_x:, :W_z] += pos_embed_z.squeeze(0)
            inps[i, :, H_x:, W_z:] += pos_embed_z.squeeze(0)
        else:
            inps[i, :, H_z:, :] += pos_embed_x.squeeze(0)
            inps[i, :, :H_z, :W_z] += pos_embed_z.squeeze(0)
            inps[i, :, :H_z, W_z:] += pos_embed_z.squeeze(0)

    return inps


def batch_add_type_embed(inps, event_inps, random_list, random_event_list, type_embedding):
    # [B, C, H, W]
    B, C, H, W = inps.shape
    H_z, W_z = 8, 8
    H_x, W_x = 16, 16
    search_image = torch.zeros(1, H_x*W_x, device=inps.device).long()
    tem_image = torch.ones(1, H_z*W_z, device=inps.device).long()
    tem_1_image = torch.ones(1, H_z*W_z, device=inps.device).long() * 2
    search_event = torch.ones(1, H_x*W_x, device=inps.device).long() * 3
    tem_event = torch.ones(1, H_z*W_z, device=inps.device).long() * 4
    tem_1_event = torch.ones(1, H_z*W_z, device=inps.device).long() * 5

    search_image = type_embedding(search_image)
    tem_image = type_embedding(tem_image)
    tem_1_image = type_embedding(tem_1_image)
    search_event = type_embedding(search_event)
    tem_event = type_embedding(tem_event)
    tem_1_event = type_embedding(tem_1_event)

    search_image = search_image.transpose(1, 2).view(1, C, H_x, W_x)
    tem_image = tem_image.transpose(1, 2).view(1, C, H_z, W_z)
    tem_1_image = tem_1_image.transpose(1, 2).view(1, C, H_z, W_z)
    search_event = search_event.transpose(1, 2).view(1, C, H_x, W_x)
    tem_event = tem_event.transpose(1, 2).view(1, C, H_z, W_z)
    tem_1_event = tem_1_event.transpose(1, 2).view(1, C, H_z, W_z)

    for i in range(len(random_list)):
        random_idx = random_list[i]
        if random_idx == 0:
            inps[i, :, :, :W_x] += search_image.squeeze(0)
            inps[i, :, :H_z, W_x:] += tem_image.squeeze(0)
            inps[i, :, H_z:, W_x:] += tem_1_image.squeeze(0)
        elif random_idx == 1:
            inps[i, :, :, W_z:] += search_image.squeeze(0)
            inps[i, :, :H_z, :W_z] += tem_image.squeeze(0)
            inps[i, :, H_z:, :W_z] += tem_1_image.squeeze(0)
        elif random_idx == 2:
            inps[i, :, :H_x, :] += search_image.squeeze(0)
            inps[i, :, H_x:, :W_z] += tem_image.squeeze(0)
            inps[i, :, H_x:, W_z:] += tem_1_image.squeeze(0)
        else:
            inps[i, :, H_z:, :] += search_image.squeeze(0)
            inps[i, :, :H_z, :W_z] += tem_image.squeeze(0)
            inps[i, :, :H_z, W_z:] += tem_1_image.squeeze(0)

    for i in range(len(random_event_list)):
        random_idx = random_event_list[i]
        if random_idx == 0:
            event_inps[i, :, :, :W_x] += search_event.squeeze(0)
            event_inps[i, :, :H_z, W_x:] += tem_event.squeeze(0)
            event_inps[i, :, H_z:, W_x:] += tem_1_event.squeeze(0)
        elif random_idx == 1:
            event_inps[i, :, :, W_z:] += search_event.squeeze(0)
            event_inps[i, :, :H_z, :W_z] += tem_event.squeeze(0)
            event_inps[i, :, H_z:, :W_z] += tem_1_event.squeeze(0)
        elif random_idx == 2:
            inps[i, :, :H_x, :] += search_event.squeeze(0)
            inps[i, :, H_x:, :W_z] += tem_event.squeeze(0)
            inps[i, :, H_x:, W_z:] += tem_1_event.squeeze(0)
        else:
            inps[i, :, H_z:, :] += search_event.squeeze(0)
            inps[i, :, :H_z, :W_z] += tem_event.squeeze(0)
            inps[i, :, :H_z, W_z:] += tem_1_event.squeeze(0)

    return inps, event_inps


def batch_recover_img(inps, random_list, random_event_list, random_id):
    # [B, C, H, W]
    B_x, C_x, H_x, W_x = inps.shape
    H_z, W_z = 8, 8
    H_x, W_x = 16, 16
    x_list = torch.zeros(B_x, C_x, H_x, W_x, device=inps.device, dtype=inps.dtype)
    z_list = torch.zeros(B_x, C_x, H_z, W_z, device=inps.device, dtype=inps.dtype)
    z_1_list = torch.zeros(B_x, C_x, H_z, W_z, device=inps.device, dtype=inps.dtype)
    x_event_list = torch.zeros(B_x, C_x, H_x, W_x, device=inps.device, dtype=inps.dtype)
    z_event_list = torch.zeros(B_x, C_x, H_z, W_z, device=inps.device, dtype=inps.dtype)
    z_1_event_list = torch.zeros(B_x, C_x, H_z, W_z, device=inps.device, dtype=inps.dtype)

    if random_id == 0: # 左右
        inps_image = inps[:, :, :H_x, :]
        inps_event = inps[:, :, H_x:, :]
    else: # 上下
        inps_image = inps[:, :, :, :W_x]
        inps_event = inps[:, :, :, W_x:]

    for i in range(len(random_list)):
        random_idx = random_list[i]
        if random_idx == 0:
            x = inps_image[i, :, :, :W_x]
            z = inps_image[i, :, :H_z, W_x:]
            z_1 = inps_image[i, :, H_z:, W_x:]
        elif random_idx == 1:
            x = inps_image[i, :, :, W_z:]
            z = inps_image[i, :, :H_z, :W_z]
            z_1 = inps_image[i, :, H_z:, :W_z]
        elif random_idx == 2:
            x = inps_image[i, :, :H_x, :]
            z = inps_image[i, :, H_x:, :W_z]
            z_1 = inps_image[i, :, H_x:, W_z:]
        else:
            x = inps_image[i, :, H_z:, :]
            z = inps_image[i, :, :H_z, :W_z]
            z_1 = inps_image[i, :, :H_z, W_z:]
        x_list[i] = x
        z_list[i] = z
        z_1_list[i] = z_1
    for i in range(len(random_event_list)):
        random_idx = random_event_list[i]
        if random_idx == 0:
            x = inps_event[i, :, :, :W_x]
            z = inps_event[i, :, :H_z, W_x:]
            z_1 = inps_event[i, :, H_z:, W_x:]
        elif random_idx == 1:
            x = inps_event[i, :, :, W_z:]
            z = inps_event[i, :, :H_z, :W_z]
            z_1 = inps_event[i, :, H_z:, :W_z]
        elif random_idx == 2:
            x = inps_event[i, :, :H_x, :]
            z = inps_event[i, :, H_x:, :W_z]
            z_1 = inps_event[i, :, H_x:, W_z:]
        else:
            x = inps_event[i, :, H_z:, :]
            z = inps_event[i, :, :H_z, :W_z]
            z_1 = inps_event[i, :, :H_z, W_z:]
        x_event_list[i] = x
        z_event_list[i] = z
        z_1_event_list[i] = z_1
    return (x_list.flatten(2), z_list.flatten(2), z_1_list.flatten(2),
            x_event_list.flatten(2), z_event_list.flatten(2), z_1_event_list.flatten(2))


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
