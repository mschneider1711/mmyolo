import torch
import torch.nn as nn
from mmengine.runner import load_checkpoint
from mmengine.logging import MMLogger
from mmyolo.models.models_biformer.biformer import BiFormer
from timm.models.layers import LayerNorm2d

from mmengine.registry import MODELS

@MODELS.register_module()
class BiFormerBackbone(BiFormer):
    def __init__(self, init_cfg=None, pretrained=None, out_indices=(0, 1, 2, 3), norm_eval=True, disable_bn_grad=False, **kwargs):
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        super().__init__(**kwargs)

        # Entferne Kopf für Dense Tasks
        del self.head
        del self.norm

        # Extra LayerNorm für die Feature-Maps
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))

        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        self.norm_eval = norm_eval
        if disable_bn_grad:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for param in m.parameters():
                        param.requires_grad = False

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'Load pretrained model from {pretrained}') 
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feat = self.extra_norms[i](x).contiguous()
            if i in self.out_indices:
                out.append(feat)
        return tuple(out)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
