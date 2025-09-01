"""
NOTE: the ConvFFN in Line should be replaced with the GatedMLP class if one want to test on lightSR
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from timm.layers.helpers import to_2tuple
from typing import Tuple

def to_3tuple(x):
    return to_2tuple(x) if isinstance(x, tuple) else (x,)*3

def index_reverse(index):
    index_r = torch.zeros_like(index)
    #ind = torch.arange(0, index.shape[-1]).to(index.device)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def ensure_last_dim_contiguous(t: torch.Tensor) -> torch.Tensor:
    """
    selective_scan_fn을 호출하기 전에 텐서를
    (B, C, L) & stride(-1)==1 로 강제 변환한다.
    비용은 copy 한 번뿐.
    """
    if t.shape[-1] == 1:            # L=1이면 어차피 OK
        return t.contiguous()
    # 1) 시퀀스 축을 맨 뒤로 보낸다
    t = t.transpose(-2, -1)
    # 2) 메모리 재배열 (row-major) --> 마지막 축 contiguous 확보
    t = t.contiguous()
    # 3) 다시 (B, C, L) 형태로 돌려놓기
    t = t.transpose(-2, -1).contiguous()
    assert t.stride(-1) == 1, f"still bad: {t.stride()}"
    return t

"""토큰 순서 재배열"""
def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

def make_target_last_sequences(seq: torch.Tensor,
                               group_ids: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seq       : (B, N, C)      – prompt 기준으로 이미 sort 된 hidden
    group_ids : (B, N)         – 동일 prompt 는 값이 동일, 연속 배치

    Returns
    -------
    rolled    : (B, N, L_max, C)   – 각 토큰이 자기 prompt 시퀀스의 마지막이 되도록
    pad_mask  : (B, N, L_max)      – True 위치만 유효, False 는 패딩
    """
    B, N, C = seq.shape
    device  = seq.device

    # --------------------------------------------------
    # 1) 그룹 구간 위치(연속 구간)와 최대 길이 L_max
    # --------------------------------------------------
    all_blocks   = []          # per-batch list of (start, end) tuples
    L_max        = 0
    for b in range(B):
        ids  = group_ids[b]
        # 변화가 생기는 인덱스 +1 이 다음 블록 시작
        bound = torch.nonzero(torch.diff(ids) != 0, as_tuple=False).flatten() + 1
        splits = torch.cat([torch.tensor([0], device=device), bound,
                            torch.tensor([N], device=device)])
        blocks = [(int(s), int(e)) for s, e in zip(splits[:-1], splits[1:])]
        all_blocks.append(blocks)
        L_max = max(L_max, max(e - s for s, e in blocks)) # 가장 긴 블록

    #print("L_max:", L_max)

    # --------------------------------------------------
    # 2) 결과 텐서 준비
    # --------------------------------------------------
    rolled   = seq.new_zeros((B, N, L_max, C))
    pad_mask = seq.new_zeros((B, N, L_max), dtype=torch.bool)

    # --------------------------------------------------
    # 3) 각 블록마다 회전 시퀀스 작성
    # --------------------------------------------------
    for b in range(B):
        for s, e in all_blocks[b]:
            block = seq[b, s:e]          # (g, C)
            g     = e - s
            for k in range(g):           # k: local index in block
                global_idx   = s + k
                shift        = -(k + 1)  # token k 가 마지막으로 오도록 회전
                rolled_seq   = torch.roll(block, shifts=shift, dims=0)
                rolled[b, global_idx, :g, :] = rolled_seq
                pad_mask[b, global_idx, :g]  = True # padding된것이 아니라 진짜 숫자가 채워진 부분을 True로 표시

    return rolled, pad_mask

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class dwconv3d(nn.Module):
    def __init__(self, hidden_features, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding = kernel_size//2, groups=hidden_features
        )
        self.act_layer = nn.GELU()

    def forward(self, x, x_size):
        # x: (B,N,C) x_size:(D,H,W)
        B,N,C = x.shape
        D,H,W = x_size
        #(B,N,C) -> (B,C,D,H,W)
        x = x.transpose(1,2).contiguous().view(B,C,D,H,W)
        x = self.act_layer(self.conv(x))
        # (B,N,C)로
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dw3 = dwconv3d(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dw3(x, x_size)
        x = self.fc2(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=2048, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj3d = nn.Conv3d(self.dim, hidden, kernel_size=1, stride=1, padding=0)
        self.CPE3d = nn.Conv3d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden)

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token, **kwargs):
        B, n, C = x.shape
        D, H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt_raw = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        #detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        detached_index = cls_policy.argmax(dim=-1)
        x_sort_values, x_sort_indices = detached_index.sort(dim=-1)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x_3d = x.permute(0, 2, 1)                        # (B, C, N)
        x_3d = x_3d.reshape(B, C, D, H, W).contiguous()  # (B, C, D, H, W)
        x_3d = self.in_proj3d(x_3d)                      # (B, hidden, D, H, W)
        x_3d = x_3d * torch.sigmoid(self.CPE3d(x_3d))    # depth-wise CPE
        hidden = x_3d.shape[1]

        x_flat = x_3d.flatten(2).transpose(1, 2).contiguous()     # (B, n, hidden)

        semantic_x = semantic_neighbor(x_flat, x_sort_indices)    # 토큰 재배치 완료
        seq, mask = make_target_last_sequences(semantic_x, x_sort_values)

        # --- 디버깅 Print 문 추가 (1) ---
        #print(f"\n[ASSM] after make_target_last_sequences | seq shape: {seq.shape}, seq stride: {seq.stride()}")

        B, N, L_max, hidden = seq.shape
        batch_seq = seq.view(-1, L_max, hidden).clone().contiguous()

        # --- 디버깅 Print 문 추가 (2) ---
        #print(f"[ASSM] after .view().clone() | batch_seq shape: {batch_seq.shape}, batch_seq stride: {batch_seq.stride()}, is_contiguous: {batch_seq.is_contiguous()}")

        batch_mask = mask.view(-1, L_max).contiguous()       # (B·N, L_max)

        prompt_seq = seq.new_zeros(B, N, L_max, self.d_state)
        valid_len = mask.sum(-1)  # (B,N)
        last_idx = (valid_len - 1).unsqueeze(-1).unsqueeze(-1)    # (B,N,1)
        prompt_seq.scatter_(2, last_idx.expand(-1, -1, 1, self.d_state),
                            prompt_raw.unsqueeze(2))
        prompt_for_scan = ensure_last_dim_contiguous(prompt_seq.view(-1, L_max, self.d_state))  # (B·N, L_max, d_state)

        y_full = self.selectiveScan(batch_seq, prompt=prompt_for_scan, mask=batch_mask)
        # 1. Get the actual sequence lengths from the mask.
        valid_lens = batch_mask.sum(dim=-1)

        # 2. Calculate the index of the last valid token (length - 1).
        #    .clamp(min=0) is a safeguard for sequences of length 0.
        last_valid_indices = (valid_lens - 1).clamp(min=0)

        # 3. Use advanced indexing to gather the last valid output of each sequence.
        num_sequences = y_full.shape[0]
        y_last = y_full[torch.arange(num_sequences, device=y_full.device), last_valid_indices]

        # 4. Reshape back to (B, N, hidden).
        y_last = y_last.view(B, N, hidden)

        y = self.out_proj(self.out_norm(y_last))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt, mask=None):
        B, L, C = x.shape

        # --- 디버깅 Print 문 추가 (3) ---
        #print(f"[forward_core] Input x | shape: {x.shape}, stride: {x.stride()}, is_contiguous: {x.is_contiguous()}")
        if mask is not None:
            #mask의 shape: (B, L) -> (B, L, 1) 로 unsqueeze하여 브로드캐스팅
            x = x*mask.unsqueeze(-1)

        prompt = prompt.contiguous()
        K = 1  # mambairV2 needs noly 1 scan
        #xs = x.permute(0, 2, 1).view(B, 1, C, L)  # B, 1, C ,L
        xs = x.transpose(1, 2).contiguous()   #

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        #xs = xs.float().view(B, -1, L).clone().contiguous()     # (B, C', L)
        xs = xs.float().view(B, -1, L)  # (B, C', L)
        xs = xs.unsqueeze(2)  # (B, C', 1, L)
        xs = xs.permute(0, 1, 3, 2).contiguous()  # (B, C', L, 1)
        xs = xs.permute(0, 1, 3, 2).squeeze(2)  # (B, C', L)

        #dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.float().view(B, -1, L).contiguous()
        Bs = Bs.float().view(B, K, -1, L).contiguous()
        Cs = (Cs.float().view(B, K, -1, L) + prompt).contiguous() # (b, k, d_state, l)  our ASE here!
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        """
        # --- 디버깅 Print 문 추가 (4) ---
        print(f"--- [forward_core] FINAL CHECK before selective_scan ---")
        print(f"--> xs | shape: {xs.shape}, stride: {xs.stride()}, is_contiguous: {xs.is_contiguous()}")
        print(f"--> dts | shape: {dts.shape}, stride: {dts.stride()}, is_contiguous: {dts.is_contiguous()}")
        """
        #print("after fix, xs.stride() =", xs.stride())
        # (기존 디버깅 프린트 직후)
        #print("stride before force copy:", xs.stride())

        # ▶ 여기에 한 줄만 추가
        xs = xs.clone(memory_format=torch.contiguous_format)
        dts = dts.clone(memory_format=torch.contiguous_format)
        Bs = Bs.clone(memory_format=torch.contiguous_format)
        Cs = Cs.clone(memory_format=torch.contiguous_format)
        Ds = Ds.clone(memory_format=torch.contiguous_format)
        As = As.clone(memory_format=torch.contiguous_format)
        dt_projs_bias = dt_projs_bias.clone(memory_format=torch.contiguous_format)
        #print("stride after final fix: ", xs.stride())

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, mask=None, **kwargs):
        x = x.contiguous()
        prompt = prompt.contiguous()
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt, mask=mask)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        """
        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        """

        self.assm = ASSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        #self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.convffn2 = ConvFFN3D(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, kernel_size=convffn_kernel_size)
        # uncomment here if you need to test on lighSR
        # self.convffn1 = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)
        # self.convffn2 =  GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):
        """ase만 사용"""
        #x_aca = self.assm(x, x_size, self.embeddingA, **params) + x
        """norm+ase+convffn"""
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA, **params) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x_aca


class BasicBlock(nn.Module):
    """ A basic ASSB for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        num_tokens (int): Token number for each token dictionary.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                AttentiveLayer(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth - 1,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class ASSB(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv', ):
        super(ASSB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, x_size, params=None):
        # x: (B, N, C), x_size=(D', H', W')
        x_out = self.residual_group(x, x_size, params)

        return x_out


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class MambaIRv2(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 d_state=8,
                 depths=(6, 6, 6, 6,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ASSB(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution


                                  [0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)

        return flops


def buildMambaIRv2Base(upscale=2):
    return MambaIRv2(img_size=64,
                     patch_size=1,
                     in_chans=3,
                     embed_dim=174,
                     d_state=16,
                     depths=(6, 6, 6, 6, 6, 6),
                     num_heads=[6, 6, 6, 6, 6, 6],
                     window_size=16,
                     inner_rank=64,
                     num_tokens=128,
                     convffn_kernel_size=5,
                     mlp_ratio=2.,
                     drop_rate=0.,
                     norm_layer=nn.LayerNorm,
                     patch_norm=True,
                     use_checkpoint=False,
                     upscale=upscale,
                     img_range=1.,
                     upsampler='pixelshuffle',
                     resi_connection='1conv')


def buildMambaIRv2_light(upscale=2):
    return MambaIRv2(img_size=64,
                     patch_size=1,
                     in_chans=3,
                     embed_dim=48,
                     d_state=8,
                     depths=(5, 5, 5, 5),
                     num_heads=[4, 4, 4, 4],
                     window_size=16,
                     inner_rank=32,
                     num_tokens=64,
                     convffn_kernel_size=5,
                     mlp_ratio=1.,
                     drop_rate=0.,
                     norm_layer=nn.LayerNorm,
                     patch_norm=True,
                     use_checkpoint=False,
                     upscale=upscale,
                     img_range=1.,
                     upsampler='pixelshuffledirect',
                     resi_connection='1conv')


def buildMambaIRv2Small(upscale=2):
    return MambaIRv2(img_size=64,
                     patch_size=1,
                     in_chans=3,
                     embed_dim=132,
                     d_state=16,
                     depths=(4, 4, 4, 4, 4, 4),
                     num_heads=[4, 4, 4, 4, 4, 4],
                     window_size=16,
                     inner_rank=64,
                     num_tokens=128,
                     convffn_kernel_size=5,
                     mlp_ratio=2.,
                     drop_rate=0.,
                     norm_layer=nn.LayerNorm,
                     patch_norm=True,
                     use_checkpoint=False,
                     upscale=upscale,
                     img_range=1.,
                     upsampler='pixelshuffle',
                     resi_connection='1conv')


def buildMambaIRv2Large(upscale=2):
    return MambaIRv2(img_size=64,
                     patch_size=1,
                     in_chans=3,
                     embed_dim=174,
                     d_state=16,
                     depths=(6, 6, 6, 6, 6, 6, 6, 6, 6),
                     num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                     window_size=16,
                     inner_rank=64,
                     num_tokens=128,
                     convffn_kernel_size=5,
                     mlp_ratio=2.,
                     drop_rate=0.,
                     norm_layer=nn.LayerNorm,
                     patch_norm=True,
                     use_checkpoint=False,
                     upscale=upscale,
                     img_range=1.,
                     upsampler='pixelshuffle',
                     resi_connection='1conv')

