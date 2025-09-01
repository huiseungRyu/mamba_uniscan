import torch
import torch.nn as nn
import torch.nn.functional as F
from model_segmamba.segmamba import GSC, MlpChannel  # 기존 GSC, MlpChannel 그대로 사용
#from model_assm_only.irv2_delete_mhsa import ASSB  # 핵심 블록만 사용
from ase_glbtoken_3d_768.ase_only_with_globaltoken import ASSB

class MambaEncoder2DScan(nn.Module):
    def __init__(self,
                 in_chans=1,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[1, 2, 3, 4],
                 window_size=8):
        super().__init__()

        dims = feat_size
        self.window_size = window_size
        self.downsample_layers = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )

        for i in range(len(dims) - 1):  # i = 0,1,2,3
            self.downsample_layers.append(nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(self._make_mamba_block(dims[i], idx=i))

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for idx in out_indices:
            norm = nn.InstanceNorm3d(dims[idx])
            self.add_module(f"norm{idx}", norm)
            self.mlps.append(MlpChannel(dims[idx], 2 * dims[idx]))

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
            num_tokens=64, #2048개로?
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

    def calculate_mask_3d(self, D, H, W, window_size):
        # 1) 전체 볼륨 인덱스별로 영역 번호 할당
        img_mask = torch.zeros((1, D, H, W, 1), dtype=torch.long)  # (1,D,H,W,1)
        d_slices = (slice(0, -window_size), slice(-window_size, -window_size // 2), slice(-window_size // 2, None))
        h_slices = d_slices;
        w_slices = d_slices
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        # 2) 3차원 패치 블록으로 분할
        mask_blocks = img_mask.unfold(1, window_size, window_size) \
            .unfold(2, window_size, window_size) \
            .unfold(3, window_size, window_size)  # (1, nD, nH, nW, window_size, window_size, window_size, 1)
        mask_blocks = mask_blocks.contiguous().view(-1, window_size ** 3)

        # 3) 어텐션 마스크 생성
        attn_mask = mask_blocks.unsqueeze(1) - mask_blocks.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
            .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask  # shape: (num_windows, window_size^3, window_size^3)


    """ ASE는 각 슬라이스별, 토큰은 GLOBAL
    def apply_mamba2d(self, x, model):
        # 한 번에 전체 볼륨 처리
        B, C, D, H, W = x.shape
        slices = torch.unbind(x, dim=2)
        embeds = [model.patch_embed(s) for s in slices]  # list of [B, N, C]
        all_embeds = torch.cat(embeds, dim=1)  # [B, D*N, C]
        global_token = all_embeds.mean(dim=1, keepdim=True)  # [B, 1, C]

        outputs = []
        for s in slices:
            x_size = (s.shape[2], s.shape[3])
            rpi = self.calculate_rpi_sa(self.window_size, s.device)
            attn_mask = self.calculate_mask(x_size[0], x_size[1], self.window_size).to(s.device)
            s_embed = model.patch_embed(s)
            params = {
                'attn_mask': attn_mask,
                'rpi_sa': rpi,
                'global_token': global_token,  # ★ 전체 볼륨 평균 토큰
            }
            s_feat = model.residual_group(s_embed, x_size, params=params)
            s_out = model.patch_unembed(s_feat, x_size)
            outputs.append(s_out.unsqueeze(2))
        return torch.cat(outputs, dim=2)
    """


    def apply_mamba2d(self, x, model):
        # 한 번에 전체 볼륨 처리
        B, C, D, H, W = x.shape

        # 1) 전체 볼륨을 “2D 이미지”로 flatten: (B, C, D*H, W)
        x_flat = x.view(B, C, D * H, W)
        x_size = (D * H, W)  # 이후 mask 계산·unembed 에 사용

        # 2) 전체-토큰 임베딩 + 글로벌 토큰 계산
        #    patch_embed: (B, C, D*H, W) -> (B, N_full, C)
        x_full_emb = model.patch_embed(x_flat)
        # 글로벌 평균 토큰: (B, 1, C)
        global_token = x_full_emb.mean(dim=1, keepdim=True)

        # 3) 동적 해상도 기반 attention mask & positional bias
        rpi = self.calculate_rpi_sa(self.window_size, x.device)
        attn_mask = self.calculate_mask(x_size[0], x_size[1], self.window_size).to(x.device)

        # 4) ASSM (residual_group) 호출
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': rpi,
            'global_token': global_token,
        }
        # x_full_emb: (B, N_full, C) -> processed tokens (B, N_full, C)
        x_full_feat = model.residual_group(x_full_emb, x_size, params=params)

        # 5) unembed 후 원래 shape 복원
        #    patch_unembed: (B, N_full, C) -> (B, C, D*H, W)
        x_out_flat = model.patch_unembed(x_full_feat, x_size)
        #    reshape to (B, C, D, H, W)
        return x_out_flat.view(B, C, D, H, W)



    def apply_mamba3d(self, x, model):
        # 3D 전용 ASE: PatchEmbed3D/UnEmbed3D 사용
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 1) 3D 패치 임베딩
        # -> (B, N, C), N = D*H*W if patch_size=1

        x_emb = model.patch_embed3d(x)
        # 2) 스테이지별 동적 해상도 (D',H',W')
        # apply_mamba3d 호출 직전 x를 downsample 했으면,
        # D'=D//ps, H'=H//ps, W'=W//ps (patch_size=ps)
        x_size3d = (D, H, W)

        # 3) mask, RPI, global_token (필요시) 계산
        #rpi = self.calculate_rpi_sa(self.window_size, x.device)
        #attn_mask = self.calculate_mask_3d(D, H, W, self.window_size).to(x.device)
        # global_token을 쓴다면
        global_token = x_emb.mean(dim=1, keepdim=True)

        params = {
                'global_token': global_token,
        }

        # 4) ASE 실행 (BasicBlock → AttentiveLayer → ASSM)
        x_feat = model.residual_group(x_emb, x_size3d, params=params)

        # 5) 3D 토큰을 다시 볼륨으로 복원
        x_out = model.patch_unembed3d(x_feat, x_size3d)  # (B, C, D, H, W)

        return x_out

    def forward_features(self, x):
        outs = []
        x = self.stem(x)
        outs.append(x)
        for stage in range(4):
            x = self.gscs[stage](x)
            x = self.apply_mamba3d(x, self.stages[stage])
            x = self.downsample_layers[stage](x)
            idx = stage + 1
            if idx in self.out_indices:
                y = getattr(self, f"norm{idx}")(x)
                mlp = self.mlps[self.out_indices.index(idx)]
                y = mlp(y)
                outs.append(y)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)
