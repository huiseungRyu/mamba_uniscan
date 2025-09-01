#!/usr/bin/env python3
#  visualize_and_save.py  ────────────────────────────────────────────
import os, warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize

# ─────────── 사용자 지정 경로 ───────────
train_dir = Path("/media/NAS/nas_187/huiseung/fullres/train")
pred_dir  = Path("/media/NAS/nas_187/huiseung/prediction_results/base_new/epoch1000_best_model")
out_dir   = Path("./compare_png")
out_dir.mkdir(exist_ok=True)

# ─────────── 색상 & 헬퍼 ───────────
CLR_GT   = (1.0, 0.0, 0.0, 0.55)   # 빨강
CLR_PRED = (0.1, 0.45, 1.0, 0.55)  # 파랑

def load_ct(stem):
    """CT/MR 3‑D 볼륨을 확장자 순서대로 로드 (.nii.gz → .npy → .npz). 없으면 None"""
    p = train_dir / f"{stem}.nii.gz"
    if p.exists():
        return nib.load(p).get_fdata()

    p = train_dir / f"{stem}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")

    p = train_dir / f"{stem}.npz"
    if p.exists():
        arr = np.load(p, mmap_mode="r")
        # 가장 흔한 키를 우선 탐색
        for k in ("image", "data", "ct", "mr"):
            if k in arr:
                return arr[k]
        # 첫 번째 배열 fallback
        return arr[arr.files[0]]

    return None

def load_gt(stem):
    p = train_dir / f"{stem}_seg.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None

def normalize_slice(slc, p_low=1, p_high=99):
    lo, hi = np.percentile(slc, [p_low, p_high])
    if hi - lo < 1e-5:
        return np.zeros_like(slc, dtype=float)
    slc = np.clip(slc, lo, hi)
    return (slc - lo) / (hi - lo)

def resize_nearest(src, tgt_shape):
    if src.shape == tgt_shape:
        return src
    return resize(src, tgt_shape, order=0,
                  preserve_range=True, anti_aliasing=False).astype(src.dtype)

def choose_slice(mask):
    return int(np.argmax(mask.sum(axis=(1,2)))) if mask.any() else mask.shape[0] // 2

# ─────────── 메인 루프 ───────────
for pred_file in pred_dir.glob("*.nii.gz"):
    stem = Path(pred_file.stem).stem          # BraTS‑GLI‑xxx‑yyy
    ct_vol = load_ct(stem)
    gt_vol = load_gt(stem)

    if ct_vol is None or gt_vol is None:
        warnings.warn(f"{stem}: CT={ct_vol is not None}, GT={gt_vol is not None}  → skip")
        continue

    pred_vol = nib.load(pred_file).get_fdata()

    # -------- 배열 차원 정리 --------
    ct_vol   = np.squeeze(ct_vol)
    gt_vol   = np.squeeze(gt_vol)
    pred_vol = np.squeeze(pred_vol)

    # 3‑D 인지 확인
    if not (ct_vol.ndim == gt_vol.ndim == pred_vol.ndim == 3):
        warnings.warn(f"{stem}: ndim mismatch CT{ct_vol.shape} GT{gt_vol.shape} P{pred_vol.shape}")
        continue

    z = choose_slice(gt_vol)
    ct_sl   = normalize_slice(ct_vol[..., z])              # 마지막 축이 Z 라 가정
    gt_sl   = gt_vol[..., z]
    pred_sl = resize_nearest(pred_vol[..., z], gt_sl.shape)

    # -------- 시각화 --------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["CT/MR", "CT + GT", "CT + Pred"]

    axes[0].imshow(ct_sl.T, cmap="gray", origin="lower")
    axes[0].set_title(titles[0]); axes[0].axis("off")

    axes[1].imshow(ct_sl.T, cmap="gray", origin="lower")
    gt_rgba = np.zeros(gt_sl.T.shape + (4,), float); gt_rgba[gt_sl.T == 1] = CLR_GT
    axes[1].imshow(gt_rgba, origin="lower")
    axes[1].set_title(titles[1]); axes[1].axis("off")

    axes[2].imshow(ct_sl.T, cmap="gray", origin="lower")
    pred_rgba = np.zeros(pred_sl.T.shape + (4,), float); pred_rgba[pred_sl.T == 1] = CLR_PRED
    axes[2].imshow(pred_rgba, origin="lower")
    axes[2].set_title(titles[2]); axes[2].axis("off")

    fig.suptitle(f"{stem} | z={z}", fontsize=15)
    plt.tight_layout()

    out_path = out_dir / f"{stem}_z{z:03d}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("✔", out_path)

print("\n🎯  모든 케이스 처리 완료 – 결과는", out_dir.resolve())
