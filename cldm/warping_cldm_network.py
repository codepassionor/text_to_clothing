import os
from os.path import join as opj
import torch
import torch.nn as nn
from einops import rearrange
from ldm.modules.attention import Normalize, CrossAttention, MemoryEfficientCrossAttention
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample
from ldm.util import exists

class CustomBasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_loss=True):
        super().__init__()
        attn_mode = "softmax-xformers" if exists(MemoryEfficientCrossAttention) else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # self-attention
        self.ff = nn.Linear(dim, dim)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_loss = use_loss

    def forward(self, x, context=None, mask=None, mask1=None, mask2=None, use_attention_mask=False,
                use_attention_tv_loss=False, tv_loss_type=None):
        # Apply standard attention if no mask or TV loss is needed
        if not (use_attention_tv_loss or use_attention_mask):
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            # Masked Attention for jewelry-specific focus
            x1, loss1 = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,
                                   mask=mask, mask1=mask1, mask2=mask2, use_attention_tv_loss=use_attention_tv_loss,
                                   tv_loss_type=tv_loss_type)
            x = x1 + x
            x2, loss2 = self.attn2(self.norm2(x), context=context, mask=mask, mask1=mask1, mask2=mask2,
                                   use_attention_tv_loss=use_attention_tv_loss, tv_loss_type=tv_loss_type)
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            loss = loss1 + loss2
            return x, loss

class CustomSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    Custom version for jewelry try-on tasks.
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_checkpoint=True, use_loss=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            CustomBasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                        disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,
                                        use_loss=use_loss) for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None, mask=None, mask1=None, mask2=None, use_attention_mask=False,
                use_attention_tv_loss=False, tv_loss_type=None):
        loss = 0
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        for block in self.transformer_blocks:
            if use_attention_tv_loss:
                x, attn_loss = block(x, context=context, mask=mask, mask1=mask1, mask2=mask2,
                                     use_attention_mask=use_attention_mask, use_attention_tv_loss=use_attention_tv_loss,
                                     tv_loss_type=tv_loss_type)
                loss += attn_loss
            else:
                x = block(x, context=context, mask=mask, mask1=mask1, mask2=mask2, use_attention_mask=use_attention_mask)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in, loss if use_attention_tv_loss else x + x_in

class JewelryWarpingModel(UNetModel):
    """
    Warping model for jewelry try-on. Incorporates depth and texture preservation.
    """

    def __init__(self, dim_head_denorm=1, use_atv_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warp_flow_blks = []
        warp_zero_convs = []

        self.encode_output_chs = [
            320, 320, 640, 640, 640, 1280, 1280, 1280, 1280
        ]
        self.encode_output_chs2 = [
            320, 320, 320, 320, 640, 640, 640, 1280, 1280
        ]

        for idx, (in_ch, cont_ch) in enumerate(zip(self.encode_output_chs, self.encode_output_chs2)):
            dim_head = in_ch // self.num_heads
            dim_head = dim_head // dim_head_denorm
            warp_flow_blks.append(CustomSpatialTransformer(
                in_channels=in_ch,
                n_heads=self.num_heads,
                d_head=dim_head,
                depth=self.transformer_depth,
                context_dim=cont_ch,
                use_checkpoint=self.use_checkpoint,
                use_loss=idx % 3 == 1,
            ))
            warp_zero_convs.append(self.make_zero_conv(in_ch))
        self.warp_flow_blks = nn.ModuleList(reversed(warp_flow_blks))
        self.warp_zero_convs = nn.ModuleList(reversed(warp_zero_convs))
        self.use_atv_loss = use_atv_loss

    def make_zero_conv(self, channels):
        return nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        mask1 = kwargs.get("mask1", None)
        mask2 = kwargs.get("mask2", None)
        loss = 0

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            hint = control.pop()

        # Process through warping blocks
        for module, warp_blk, warp_zc in zip(self.output_blocks[3:], self.warp_flow_blks, self.warp_zero_convs):
            hint = control.pop()
            h, attn_loss = self.warp(h, hint, warp_blk, warp_zc, mask1=mask1, mask2=mask2)
            loss += attn_loss
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        for module in self.output_blocks[:3]:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return h, loss

    def warp(self, x, hint, crossattn_layer, zero_conv, mask1=None, mask2=None):
        hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
        if self.use_atv_loss:
            output, attn_loss = crossattn_layer(x, hint, mask1=mask1, mask2=mask2, use_attention_tv_loss=True)
            output = zero_conv(output)
            return output + x, attn_loss
        else:
            output = crossattn_layer(x, hint)
            output = zero_conv(output)
            return output + x, 0
