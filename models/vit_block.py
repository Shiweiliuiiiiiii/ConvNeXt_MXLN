import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.layers import DropPath
from timm.models.layers import Mlp, PatchEmbed

from functools import partial

class CustomBlock(TimmBlock):
    
    def __init__(self, *args, pre_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(x))
            x = self.norm1(x)
            x = x + self.drop_path(self.mlp(x))
            x = self.norm2(x)
        return x


class CustomVisionTransformer(TimmVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            CustomBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                pre_norm=True,
                # pre_norm=False if i < depth // 6 else True
            )
            for i in range(depth)
        ])

# model = CustomVisionTransformer(
#     patch_size=16, embed_dim=192, depth=12, num_heads=3
# )