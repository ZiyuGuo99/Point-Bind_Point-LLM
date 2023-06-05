'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

# Modified from github.com/openai/CLIP
from collections import OrderedDict
import ipdb
from PIL import Image
from imagebind.imagebind_model import ModalityType
from torchvision import transforms
from imagebind import imagebind_model

import timm
from torch import nn
from data.dataset_3d import  *

from torch.nn.parameter import Parameter
from easydict import EasyDict

from models.pointbert.point_encoder import PointTransformer_BIND
from models.i2p_mae.I2P_MAE import I2P_MAE_BIND
POINTBERT_CONFIG = './models/pointbert/PointTransformer_8192point.yaml'

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class POINTBIND(nn.Module):
    def __init__(self, point_encoder, pc_feat_dims):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.bind = imagebind_model.imagebind_huge().eval().cuda()

        self.point_encoder = point_encoder
        self.pc_projection = nn.Parameter(torch.empty(pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):
        inputs = {
            ModalityType.TEXT: text.squeeze().cuda(),
            ModalityType.VISION: image,
        }
        with torch.no_grad():
            embeddings = self.bind(inputs)
        image_embed = embeddings[ModalityType.VISION]
        text_embed_all = embeddings[ModalityType.TEXT]

        pc_embed = self.encode_pc(pc)
        pc_embed = self.bind.modality_head_point(pc_embed)
        pc_embed = self.bind.modality_postprocessor_point(pc_embed)

        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

def PointBind_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    config = cfg_from_yaml_file(POINTBERT_CONFIG)
    point_encoder = PointTransformer_BIND(config.model, args=args)
    model = POINTBIND(point_encoder, pc_feat_dims=768)
    return model

def PointBind_I2PMAE(args=None):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    point_encoder = I2P_MAE_BIND()
    model = POINTBIND(point_encoder, pc_feat_dims=384)
    return model
