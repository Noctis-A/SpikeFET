"""
Basic spikefet model.
"""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.spikefet.spike_models_tiny import SpikeFormer_tiny
from lib.models.spikefet.spike_models_base import SpikeFormer_base
# from lib.models.spikefet.vit_ce_BACKUPS import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SpikeFET(nn.Module):
    """ This is the base class for spikefet """

    def __init__(self, transformer, box_head_image, box_head_event, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head_image = box_head_image
        self.box_head_event = box_head_event

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head_image.feat_sz)
            self.feat_len_s = int(box_head_image.feat_sz ** 2)

        if self.aux_loss:
            self.box_head_image = _get_clones(self.box_head_image, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                event_template: torch.Tensor,        # torch.Size([4, 1, 19, 10000])
                event_search: torch.Tensor,          # torch.Size([4, 1, 19, 10000])
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        # before feeding into backbone, we need to concat four vectors, or two two concat;
        x, temp = self.backbone(z=template, x=search, event_z=event_template, event_x=event_search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        # if isinstance(x, list):
        #     feat_last = x[-1]
        out = self.forward_head_image(feat_last[0], None)
        out_1 = self.forward_head_event(feat_last[1], None)
        # out.update(attn)
        out['backbone_feat'] = x
        out['similarity'] = temp
        return [out, out_1]

    def forward_head_image(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # output the last 256
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)  768*256
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        ## dual head   768+768)*256
        if cat_feature.shape[1] == 320:
            enc_opt = cat_feature[:, -self.feat_len_s:]
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        else:
            enc_opt1 = cat_feature[:, -self.feat_len_s:] # img
            enc_opt2 = cat_feature[:, :self.feat_len_s] # event
            enc_opt = torch.cat([enc_opt1, enc_opt2], dim=-1)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head_image(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head_image(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_head_event(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # output the last 256
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)  768*256
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        ## dual head   768+768)*256
        if cat_feature.shape[1] == 320:
            enc_opt = cat_feature[:, -self.feat_len_s:]
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        else:
            enc_opt1 = cat_feature[:, -self.feat_len_s:] # img
            enc_opt2 = cat_feature[:, :self.feat_len_s] # event
            enc_opt = torch.cat([enc_opt1, enc_opt2], dim=-1)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head_event(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head_event(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_spikefet(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('SpikeFET' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'SpikeFormer':
        backbone = SpikeFormer(pretrained)

        hidden_dim = backbone.embed_dim[-1]

    elif cfg.MODEL.BACKBONE.TYPE == 'SpikeFormer_tiny':
        backbone = SpikeFormer_tiny(pretrained)

        hidden_dim = backbone.embed_dim[-1]

    elif cfg.MODEL.BACKBONE.TYPE == 'SpikeFormer_base':
        backbone = SpikeFormer_base(pretrained)

        hidden_dim = backbone.embed_dim[-1]

    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head_image = build_box_head(cfg, hidden_dim, type=cfg.MODEL.BACKBONE.TYPE)
    box_head_event = build_box_head(cfg, hidden_dim, type=cfg.MODEL.BACKBONE.TYPE)

    model = SpikeFET(
        backbone,
        box_head_image,
        box_head_event,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'SpikeFET' in cfg.MODEL.PRETRAIN_FILE and training:
        from collections import OrderedDict
        import re
        from lib.models.spikefet.utils import revise_keys_finetune
        checkpoint = torch.load(os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE), map_location="cpu")
        state_dict = checkpoint["net"]

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            replaced = False
            for pattern, replacements in revise_keys_finetune:
                if re.match(pattern, key):
                    for r in replacements:
                        new_key = re.sub(pattern, r, key)
                        new_state_dict[new_key] = value
                        replaced = True
            if not replaced:
                new_state_dict[key] = value
        state_dict = new_state_dict
        state_dict['backbone.type_embedding.weight'][3:] = state_dict['backbone.type_embedding.weight'][:3]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
