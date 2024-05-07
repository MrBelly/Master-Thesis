from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Channel_Attention_Block(nn.Module):
    """
    Channel Attention Block 
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
                                      
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm = LayerNorm(self.dim, eps=1e-6)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)



    def forward(self, x):
        input_ = x
        
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        
        x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_ + self.drop_path(x)
        
       

        return x



class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')
        
        
        
        self.convs = OrderedDict()
        
        
        
        #  Upconv i, 1 = Channel Attentio Block with Embedding      
        
        #self.convs[("upconv", 0, 0)] = Channel_Attention_Block(dim = 64, drop = 0.20000000298023224)
                                      
        #self.convs[("upconv", 1, 0)] = Channel_Attention_Block(dim = 40, drop = 0.08235294371843338)
        
        self.convs[("upconv", 0, 0)] = Channel_Attention_Block(dim = 64, drop_path = 0.20000000298023224)
                                      
        self.convs[("upconv", 1, 0)] = Channel_Attention_Block(dim = 40, drop_path =0.20000000298023224)
                                      
        # Upconv i, 1 = conv
        
        self.convs[("upconv", 0, 1)] = ConvBlock(112, 64)
        self.convs[("upconv", 1, 1)] = ConvBlock(64,40)  
        self.convs[("upconv", 2, 1)] = ConvBlock(40,24) 
        
        # dispconv i = conv3x3
        
        self.convs[("dispconv", 0)] = Conv3x3(64, 1)
        self.convs[("dispconv", 1)] = Conv3x3(40, 1)
        self.convs[("dispconv", 2)] = Conv3x3(6, 1)

        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
        
        
        
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        
        
        
        ## First Part
        
        ##****************************************************************
        # Getting the features from the encoder
        x_40 = input_features[2]
        x_80 = input_features[1]
        x_160 = input_features[0]

        

        # Upsampling with reshaping for decreasing the complexity
        x_40_depth =F.pixel_shuffle(x_40, 2)
        

        
        # Concetating upsampled x_40 with x_80
        x = torch.cat((x_40_depth,x_80),dim=1)
        
        

        
        x = self.convs[("upconv", 0, 1)](x)

        x = self.convs[("upconv", 0, 0)](x)

        
        
        # Extracting the low dimension depth head
        
        x_1 = self.convs[("dispconv", 0)](x)

        
        out_1 = self.sigmoid(upsample(x_1, mode='bilinear'))
        
        self.outputs[("disp", 2)] = out_1
        


        
        ##****************************************************************
        
        # Second Part
        
        # Upsampling with reshaping for decreasing the complexity

        
        x = F.pixel_shuffle(x, 2)

        
        # Concetating upsampled x with x_160

        
        x = torch.cat((x_160,x),dim=1)


        
        # 2nd Channel-wise Attention Module
        
        
        
        
        
        x = self.convs[("upconv", 1, 1)](x)

        x = self.convs[("upconv", 1, 0)](x)


                
        x_2 = self.convs[("dispconv", 1)](x)
        
        out_2 = self.sigmoid(upsample(x_2, mode='bilinear'))
        
        self.outputs[("disp", 1)] = out_2
        

        
        
        #*************************************
        
        # Third Part
        
        x = self.convs[("upconv", 2, 1)](x)

  
        x = F.pixel_shuffle(x, 2)

        x_3 = self.convs[("dispconv", 2)](x)
        
        out_3 = self.sigmoid(upsample(x_3, mode='bilinear'))
        
        self.outputs[("disp", 0)] = out_3
        
    
        

        return self.outputs

