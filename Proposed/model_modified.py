# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
from Proposed.asse.ase_only import ASSB
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v1",
            nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out


# <<< SemanticMambaLayer 수정: 스테이지별 하이퍼파라미터를 받도록 __init__ 변경 >>>
class SemanticMambaLayer(nn.Module):
    def __init__(self, dim, d_state, depth, num_heads, inner_rank, num_tokens, window_size=16):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.window_size = window_size

        # <<< ASSB 초기화 시, 전달받은 스테이지별 파라미터를 사용 >>>
        self.mamba = ASSB(
            dim=dim,
            d_state=d_state,
            idx=None,
            input_resolution=(1, 1),  # placeholder
            depth=depth,
            num_heads=num_heads,
            window_size=self.window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=5,
            mlp_ratio=2.,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            img_size=64,  # placeholder
            patch_size=1,  # placeholder
            resi_connection='1conv',
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_size3d = (D, H, W)
        x_mamba = self.mamba.residual_group(x_norm, x_size3d, params={})

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# <<< MambaEncoder 수정: 스테이지별 하이퍼파라미터 리스트를 받도록 __init__ 변경 >>>
class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 d_states=[16, 16, 16, 16], num_tokens_list=[2048, 1024, 512, 256],
                 inner_rank_list=[128, 128, 64, 64], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=4, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]  # This seems related to another Mamba type, keeping as is
        cur = 0

        # <<< 스테이지 생성 루프 수정: 각 스테이지에 맞는 하이퍼파라미터를 전달 >>>
        for i in range(4):
            # SemanticMambaLayer를 여러 번 감싸지 않고, depth를 직접 전달
            stage = SemanticMambaLayer(
                dim=dims[i],
                d_state=d_states[i],
                depth=depths[i],  # BasicBlock이 내부적으로 루프를 돌 depth
                num_heads=4,  # Placeholder, ASSB/AttentiveLayer에서 사용
                inner_rank=inner_rank_list[i],
                num_tokens=num_tokens_list[i]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


# <<< SegMamba 수정: 계층적 하이퍼파라미터를 정의하고 MambaEncoder에 전달 >>>
class SegMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            # <<< 계층적 하이퍼파라미터 리스트 정의 >>>
            d_states_list=[16, 16, 16, 16],
            num_tokens_list=[2048, 1024, 512, 256],
            inner_rank_list=[128, 128, 64, 64],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims

        # <<< MambaEncoder 초기화 시, 계층적 하이퍼파라미터 리스트 전달 >>>
        self.vit = MambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            d_states=d_states_list,
            num_tokens_list=num_tokens_list,
            inner_rank_list=inner_rank_list,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.encoder0 = nn.Conv3d(self.in_chans, self.feat_size[0], kernel_size=1)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.in_chans, out_channels=self.feat_size[0],
            kernel_size=3, stride=2, norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[1],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[2],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[3],
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.hidden_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.feat_size[2],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[1],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[0],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3, in_channels=self.feat_size[0], out_channels=self.feat_size[0],
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc0 = self.encoder0(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        return self.out(out)
