# softsort.py  (기존 코드 아래에 그대로 추가) ───────────────────────────
import torch.nn.functional as F
import torch
from torch import Tensor


class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


class SoftSort_p1(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(SoftSort_p1, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(SoftSort_p2, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = ((scores.transpose(1, 2) - sorted) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

class RefSliceSoftSort(torch.nn.Module):
    """
    Memory‑efficient SoftSort:
      ‑ 전체 토큰 n,  참조열 m (m << n, 예: 4096)
      ‑ pairwise 행렬을 (n × m) 만 생성
    Args:
        m (int):  reference column size
        tau (float): temperature
        hard (bool): STE‑argmax permutation
    """
    def __init__(self, m: int = 4096, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.m, self.tau, self.hard = m, tau, hard

    @torch.no_grad()
    def _build_perm(self, P_hat: torch.Tensor, ref_idx: torch.Tensor):
        """
        P_hat: (B, n_slice, m)
        ref_idx: (B, m)  전역 Top‑m 인덱스
        Return: (B, n_slice)  전역 permutation 인덱스
        """
        # 각 행 argmax → ref 열의 순위 (0‥m‑1)
        col_rank = P_hat.argmax(-1)                              # (B, n_slice)
        # ref_idx 에서 실제 전역 인덱스 취득
        return torch.gather(ref_idx, 1, col_rank)

    def forward(self, scores: torch.Tensor):
        """
        scores: (B, n) 전역 importance score
        Return:  (B, n)  permutation 인덱스 (0‥n‑1)
        """
        B, n = scores.shape
        m = min(self.m, n)
        # 1) 전역 Top‑m score와 인덱스
        topk = torch.topk(scores, m, dim=1, sorted=True)
        ref_val, ref_idx = topk.values, topk.indices            # (B, m)

        # 2) 슬라이스 단위 SoftSort
        slice_len = 4096
        perm_parts = []
        for idx in torch.split(torch.arange(n, device=scores.device), slice_len):
            slice_scores = scores[:, idx]                       # (B, slice_len)
            # pairwise (B, slice_len, m)
            pairwise = (slice_scores.unsqueeze(-1) - ref_val.unsqueeze(1)).abs().neg() / self.tau
            P_hat = F.softmax(pairwise, dim=-1)                 # (B, slice_len, m)

            if self.hard:
                hard_mask = torch.zeros_like(P_hat)
                hard_mask.scatter_(-1, P_hat.topk(1, -1)[1], 1.)
                P_hat = (hard_mask - P_hat).detach() + P_hat

            # 3) slice‑별 전역 permutation 인덱스
            perm_slice = self._build_perm(P_hat, ref_idx)       # (B, slice_len)
            perm_parts.append(perm_slice)

        perm = torch.cat(perm_parts, dim=1)                     # (B, n)
        return perm
# ───────────────────────────────────────────────────────────────────────
