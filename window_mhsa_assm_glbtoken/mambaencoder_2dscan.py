import torch
import torch.nn as nn
import torch.nn.functional as F
from model_segmamba.segmamba import GSC, MlpChannel  # 기존 GSC, MlpChannel 그대로 사용
from window_mhsa_assm_glbtoken.mambairv2 import ASSB  # 핵심 블록만 사용


class MambaEncoder2DScan(nn.Module):
    def __init__(self,
                 in_chans=1,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 window_size=8):
        super().__init__()

        dims = feat_size
        self.window_size = window_size
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(4):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(self._make_mamba_block(dims[i], idx=i))

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            norm = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f"norm{i_layer}", norm)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def _make_mamba_block(self, dim, idx):
        block = ASSB(
            dim=dim,
            d_state=16,
            idx=idx,
            input_resolution=(1, 1),  # placeholder, 실제로는 런타임에 갱신
            depth=2,
            num_heads=4,
            window_size=self.window_size,
            inner_rank=32,
            num_tokens=64,
            convffn_kernel_size=5,
            mlp_ratio=2.,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            img_size=64,
            patch_size=1,
            resi_connection='1conv',
        )
        return block

    def calculate_rpi_sa(self, window_size, device):
        coords_h = torch.arange(window_size, device=device)
        coords_w = torch.arange(window_size, device=device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, H, W, window_size):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -window_size), slice(-window_size, -window_size // 2), slice(-window_size // 2, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -window_size // 2), slice(-window_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = img_mask.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
        mask_windows = mask_windows.contiguous().view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def apply_mamba2d(self, x, model):
        B, C, D, H, W = x.shape
        slices = torch.unbind(x, dim=2)
        outputs = []
        for s in slices:
            x_size = (s.shape[2], s.shape[3])
            rpi = self.calculate_rpi_sa(self.window_size, s.device)
            attn_mask = self.calculate_mask(x_size[0], x_size[1], self.window_size).to(s.device)
            s_embed = model.patch_embed(s)
            s_feat = model.residual_group(s_embed, x_size, params={
                'attn_mask': attn_mask,
                'rpi_sa': rpi
            })
            s_out = model.patch_unembed(s_feat, x_size)
            outputs.append(s_out.unsqueeze(2))
        return torch.cat(outputs, dim=2)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.apply_mamba2d(x, self.stages[i])
            if i in self.out_indices:
                norm = getattr(self, f"norm{i}")
                out = norm(x)
                out = self.mlps[i](out)
                outs.append(out)
        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)
