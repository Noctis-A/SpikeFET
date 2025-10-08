from functools import partial
import torch
import random
import re
from spikingjelly.clock_driven import layer
from timm.models.layers import DropPath
import torch.nn as nn
from lib.models.spikefet.utils import revise_keys
from lib.models.spikefet.utils import (add_modal_embed, batch_combine_img,
                                          batch_add_pos_embed, batch_add_type_embed, batch_recover_img)
from collections import OrderedDict

# timestep 1x8
T = 8


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=8):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(torch.clamp(input, 0, lens) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, spike=multispike, norm=8):
        super().__init__()
        self.lens = norm
        self.spike = spike
        self.norm = norm

    def forward(self, inputs):
        return self.spike.apply(inputs) / self.norm


def MS_conv_unit(in_channels, out_channels, kernel_size=1, padding=0, groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(out_channels)
        )
    )


class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            bias=False,
            kernel_size=7,
            padding=3,

    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = Multispike()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = Multispike()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = Multispike()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.spike1(x)
        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike2(x)
        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.spike3(x)
        x = self.pwconv2(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,

    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = Multispike()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = Multispike()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)

        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)

        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = Multispike()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = Multispike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x)

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class LePEAttention(nn.Module):
    def __init__(self, dim=None, resolution=None, idx=-1, split_num=2, dim_out=None, num_heads=8, attn_drop=0.,
                 proj_drop=0., qk_scale=None):
        super().__init__()
        self.split_num = split_num
        self.num_heads = num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale
        if idx == -1:
            H_num, W_num = 1, 1
        elif idx == 0:
            H_num, W_num = 1, self.split_num
        elif idx == 1:
            H_num, W_num = self.split_num, 1
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.H_num = H_num
        self.W_num = W_num

    def forward(self, q, k, v, v_lamda):
        """
        q,k,v: T, B C H W
        """

        T, B, C, H, W = q.shape
        self.H_sp = H // self.H_num
        self.W_sp = W // self.W_num

        q = (
            q.reshape(T, B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
                .permute(0, 1, 3, 5, 4, 6, 2)
                .reshape(T, -1, self.H_sp * self.W_sp, C)
                .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        k = (
            k.reshape(T, B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
                .permute(0, 1, 3, 5, 4, 6, 2)
                .reshape(T, -1, self.H_sp * self.W_sp, C)
                .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        v = (
            v.reshape(T, B, C * v_lamda, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp)
                .permute(0, 1, 3, 5, 4, 6, 2)
                .reshape(T, -1, self.H_sp * self.W_sp, C * v_lamda)
                .reshape(T, -1, self.H_sp * self.W_sp, self.num_heads, C * v_lamda // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = (
            x.transpose(2, 3).reshape(T, -1, self.H_sp * self.W_sp, C * v_lamda)
                .reshape(T, B, H // self.H_sp, W // self.W_sp, self.H_sp, self.W_sp, -1)
                .permute(0, 1, 2, 4, 3, 5, 6)
                .reshape(T, B, H, W, -1)
                .permute(0, 1, 4, 2, 3)
                .contiguous()# T, B, C, H, W
        )

        return x

class MS_Attention_linear_cswin(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            last_stage=False,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        if last_stage == False:
            self.branch_num = 2
        else:
            self.branch_num = 1

        self.dim = dim
        self.num_heads = num_heads // self.branch_num
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = Multispike()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = Multispike()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = Multispike()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = Multispike()

        self.attn_spike = Multispike()

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    idx=-1, num_heads=num_heads, qk_scale=self.scale)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    idx=i, num_heads=num_heads // 2, qk_scale=self.scale)
                for i in range(self.branch_num)])

    def forward(self, x):
        T, B, C, H, W = x.shape
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C_v, H, W)

        q = self.q_spike(q)
        k = self.q_spike(k)
        v = self.q_spike(v)

        if self.branch_num == 2:
            x1 = self.attns[0](q[:, :, :C // 2, :, :], k[:, :, :C // 2, :, :], v[:, :, :C_v // 2, :, :],
                               self.lamda_ratio)
            x2 = self.attns[1](q[:, :, C // 2:, :, :], k[:, :, C // 2:, :, :], v[:, :, C_v // 2:, :, :],
                               self.lamda_ratio)
            x = torch.cat([x1, x2], dim=2)
        else:
            x = self.attns[0](q, k, v, self.lamda_ratio)

        x = self.attn_spike(x)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        return x
class MS_Block_cswin(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            init_values=1e-6,
            last_stage=False,
            resolution=14,
            T=None
    ):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        self.attn = MS_Attention_linear_cswin(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
            last_stage=last_stage,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        x = x + self.drop_path(self.conv(x) * self.layer_scale1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        x = x + self.drop_path(self.attn(x) * self.layer_scale2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        x = x + self.drop_path(self.mlp(x) * self.layer_scale3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        return x


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,

    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x


class Efficient_Spiking_Transformer(nn.Module):
    def __init__(self, T=1,
                 choice=None,
                 img_size_h=224,
                 img_size_w=224,
                 patch_size=16,
                 embed_dim=[128, 256, 512, 640],
                 num_heads=8,
                 mlp_ratios=4,
                 in_channels=3,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 num_classes=1000,
                 qkv_bias=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),  # norm_layer=nn.LayerNorm shaokun
                 depths=8,
                 sr_ratios=1,
                 mlp_ratio=4.,
                 nb_classes=1000,
                 kd=True):
        super().__init__()

        ### MAE encoder spikformer
        self.T = T
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule
        self.downsample1_1_images = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,

        )

        self.downsample1_1_events = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,

        )

        self.ConvBlock1_1_images = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock1_1_events = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2_images = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.downsample1_2_events = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2_images = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock1_2_events = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2_images = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.downsample2_events = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1_images = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_1_events = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2_images = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2_events = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3_images = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.downsample3_events = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_cswin(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,

                )
                for j in range(int(depths * 0.75))
            ]
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_cswin(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j+int(depths * 0.75)],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    last_stage=True,
                )
                for j in range(int(depths * 0.25))
            ]
        )

        self.pos_embed_z = nn.Parameter(torch.zeros(1, embed_dim[2], 8, 8))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, embed_dim[2], 16, 16))
        self.modal_embedding = nn.Embedding(2, embed_dim[2])
        self.type_embedding = nn.Embedding(6, embed_dim[2])
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        torch.nn.init.xavier_uniform_(self.pos_embed_z)
        torch.nn.init.xavier_uniform_(self.pos_embed_x)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature_images(self, x):
        x = self.downsample1_1_images(x)

        for blk in self.ConvBlock1_1_images:
            x = blk(x)

        x = self.downsample1_2_images(x)

        for blk in self.ConvBlock1_2_images:
            x = blk(x)

        x = self.downsample2_images(x)

        for blk in self.ConvBlock2_1_images:
            x = blk(x)

        for blk in self.ConvBlock2_2_images:
            x = blk(x)

        x = self.downsample3_images(x)

        return x

    def forward_feature_events(self, event_x):
        event_x = self.downsample1_1_events(event_x)

        for blk in self.ConvBlock1_1_events:
            event_x = blk(event_x)

        event_x = self.downsample1_2_events(event_x)

        for blk in self.ConvBlock1_2_events:
            event_x = blk(event_x)

        event_x = self.downsample2_events(event_x)

        for blk in self.ConvBlock2_1_events:
            event_x = blk(event_x)

        for blk in self.ConvBlock2_2_events:
            event_x = blk(event_x)

        event_x = self.downsample3_events(event_x)

        return event_x

    def forward_fusion(self, x, event_x, random_list, random_event_list, random_id):
        x, event_x = x.flatten(0, 1), event_x.flatten(0, 1)

        x = batch_add_pos_embed(x, random_list, self.pos_embed_x, self.pos_embed_z)
        event_x = batch_add_pos_embed(event_x, random_event_list, self.pos_embed_x, self.pos_embed_z)

        x, event_x = batch_add_type_embed(x, event_x, random_list, random_event_list, self.type_embedding)
        x, event_x = add_modal_embed(x, event_x, self.modal_embedding, random_id)
        if random_id == 0:
            x = torch.cat([x, event_x], dim=2)
        else:
            x = torch.cat([x, event_x], dim=3)

        B, C, H, W = x.shape

        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        for blk in self.block3:
            x = blk(x)

        for blk in self.block4:
            x = blk(x)

        x = x.reshape(self.T, B, C, H, W)

        return x.flatten(0, 1)

    def forward(self, z, x, event_z, event_x, **kwargs
                ):
        random_id = random.choice(list(range(2)))
        x, random_list = batch_combine_img(z, x, random_id)
        event_x, random_event_list = batch_combine_img(event_z, event_x, random_id)

        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        event_x = (event_x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        x = self.forward_feature_images(x)
        event_x = self.forward_feature_events(event_x)

        x = self.forward_fusion(x, event_x, random_list, random_event_list, random_id)

        x, z, z_1, event_x, event_z, event_z_1 = batch_recover_img(x, random_list, random_event_list, random_id)

        temp = [(z + event_z) / 2, (z_1 + event_z_1) / 2]

        x = torch.cat((z.transpose(1, 2), x.transpose(1, 2)), dim=1)
        event_x = torch.cat((event_z.transpose(1, 2), event_x.transpose(1, 2)), dim=1)

        return [x, event_x], temp


def Efficient_Spiking_Transformer_L(**kwargs):
    model = Efficient_Spiking_Transformer(
        T=1,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128,256,512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1,
        **kwargs)
    return model


def Efficient_Spiking_Transformer_H(**kwargs):
    model = Efficient_Spiking_Transformer(
        T=1,
        choice="large",
        img_size_h=32,
        img_size_w=32,
        patch_size=16,
        embed_dim=[196, 384, 768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        **kwargs)
    return model


def SpikeFormer_base(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model = Efficient_Spiking_Transformer_L(**kwargs)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        state_dict = checkpoint["model"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    return model

def SpikeFormer_large(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model = Efficient_Spiking_Transformer_H(**kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            replaced = False
            for pattern, replacements in revise_keys:
                if re.match(pattern, key):
                    for r in replacements:
                        new_key = re.sub(pattern, r, key)
                        new_state_dict[new_key] = value
                        replaced = True
            if not replaced:
                new_state_dict[key] = value
        state_dict = new_state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    return model

if __name__ == "__main__":
    model = Efficient_Spiking_Transformer_L()

    print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
