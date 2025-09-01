from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torchsort


# -----------------------------------------------------------------------------
# Utility modules
# -----------------------------------------------------------------------------
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

def reorder_tokens_by_soft_rank(x, importance, regularization=1.0):
    """
    x           : (B, N, D)  — 정렬 대상 토큰
    importance  : (B, N)     — 토큰별 importance score
    returns y   : (B, N, D)  — importance 기준 정렬된 토큰 시퀀스
    """
    B, N, D = x.shape
    # 1) soft rank (연속 순위) — [1..N] 범위, differentiable
    soft_ranks = torchsort.soft_rank(
        importance, regularization_strength=regularization
    )                           # (B, N)
    # 2) 정수/소수 부분 분리
    rank_floor = torch.clamp(soft_ranks.floor().long(), 0, N - 1)       # (B, N)
    frac = (soft_ranks - rank_floor.float())                            # (B, N)

    # 3) 두 위치에 선형 보간 분배
    y = x.new_zeros(B, N, D)

    # 첫 번째 위치(가중치 = 1-frac)
    y.scatter_add_(
        dim=1,
        index=rank_floor.unsqueeze(-1).expand(-1, -1, D),   # (B, N, D)
        src=x * (1.0 - frac).unsqueeze(-1)
    )

    # 두 번째 위치(가중치 = frac) — rank_floor+1 이 범위를 넘지 않도록 마스크
    mask = (rank_floor + 1 < N)
    valid_idx = (rank_floor + 1).clamp(max=N-1)
    y.scatter_add_(
        dim=1,
        index=valid_idx.unsqueeze(-1).expand(-1, -1, D),
        src=(x * frac.unsqueeze(-1)) * mask.unsqueeze(-1)
    )

    return y

# -----------------------------------------------------------------------------
# Core MambaLayer with Demystifying‑style importance
# -----------------------------------------------------------------------------
class MambaLayer(nn.Module):
    """Single‑scan Mamba layer + token re‑ordering.

    * Importance score pipeline follows Demystifying‑the‑Token‑Dynamics (in_proj →
      depth‑wise conv → x_proj → dt_proj → importance).
    * Reordering with SoftSort (full) or RefSliceSoftSort (Top‑k memory‑efficient).
    """

    def __init__(
            self,
            dim: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            num_slices: int | None = None,
            dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # Demystifying parameters ------------------------------------------------
        self.expand = expand
        self.d_inner = dim * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank

        # Projections ------------------------------------------------------------
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        self.conv1d_x = nn.Conv1d(self.d_inner // 2, self.d_inner // 2, kernel_size=d_conv,
                                  padding="same", groups=self.d_inner // 2, bias=True)
        self.conv1d_z = nn.Conv1d(self.d_inner // 2, self.d_inner // 2, kernel_size=d_conv,
                                  padding="same", groups=self.d_inner // 2, bias=True)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)
        self.importance = nn.Linear(self.d_inner // 2, 1, bias=False)

        # Sorter ----------------------------------------------------------------

        # Uniscan Mamba ----------------------------------------------------------
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v1",  # uniscan
            nslices=num_slices,
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, D, H, W) or similar
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        N = math.prod(spatial_shape)

        x_skip = x
        x_tok = x.reshape(B, C, N).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm(x_tok)

        # Demystifying importance pipeline -------------------------------------
        xz = self.in_proj(x_norm)  # (B, N, d_inner)
        xz = xz.transpose(1, 2)  # (B, d_inner, N)
        x_part, z_part = xz.chunk(2, dim=1)  # each (B, d_inner/2, N)

        x_part = F.silu(self.conv1d_x(x_part))
        z_part = F.silu(self.conv1d_z(z_part))

        x_dbl = self.x_proj(x_part.transpose(1, 2))  # (B, N, dt_rank + 2*d_state)
        dt, _, _ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, N, d_inner/2)

        scores = self.importance(dt).squeeze(-1)  # (B, N)

        # Sorting ---------------------------------------------------------------
        x_sorted = reorder_tokens_by_soft_rank(x_norm, scores, regularization=1.0)

        # Mamba + skip ----------------------------------------------------------
        out = self.mamba(x_sorted)  # (B, N, C)
        out = out.transpose(1, 2).reshape(B, C, *spatial_shape) + x_skip
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


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
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
            x = self.gscs[i](x)
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


class SegMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
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
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
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
        out = self.decoder1(dec0)

        return self.out(out)
