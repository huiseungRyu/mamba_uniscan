#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ─── mask_to_rgba 정의 ───
def mask_to_rgba(mask, color):
    """
    2D mask 배열(mask.shape == (H, W), dtype=float or int)을
    RGBA 이미지로 변환합니다.
    """
    bool_mask = mask.astype(bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[bool_mask] = color
    return rgba

# ─── convert_labels 정의 ───
def convert_labels(labels):
    """
    입력: labels (torch.Tensor, shape=(Z, H, W), int 라벨맵)
    출력: torch.Tensor, shape=(3, Z, H, W), float 바이너리 마스크
      [0]=WT, [1]=TC, [2]=ET
    """
    labels = labels.unsqueeze(dim=0)  # (1, Z, H, W)
    result = [
        (labels == 1) | (labels == 2) | (labels == 3),  # WT
        (labels == 1) | (labels == 3),                  # TC
        (labels == 3)                                  # ET

    ]
    return torch.cat(result, dim=0).float()           # (3, Z, H, W)

def main():
    stem         = "BraTS-GLI-00469-000"
    raw_data_dir = "/media/NAS/nas_187/huiseung/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    pred_root    = "/media/NAS/nas_187/huiseung/prediction_results/ase_only_with_globaltoken/epoch1000_best_model"#base_new/epoch1000_best_model"
    out_dir      = Path("/media/NAS/nas_187/huiseung/visualizations/0526_ours") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Input image (T1ce) 불러오기 ───
    IMG_PATH = Path(raw_data_dir) / stem / "t1c.nii.gz"
    raw_img  = sitk.GetArrayFromImage(sitk.ReadImage(str(IMG_PATH))).astype(np.float32)
    # shape = (Z, H, W)

    # ─── GT 원본 (Z, H, W) 불러오기 ───
    GT_PATH = Path(raw_data_dir) / stem / "seg.nii.gz"
    raw_gt  = sitk.GetArrayFromImage(sitk.ReadImage(str(GT_PATH))).astype(np.int32)

    # ─── GT 바이너리 마스크 (3, Z, H, W) 생성 ───
    gt_masks = convert_labels(torch.from_numpy(raw_gt)).numpy()

    # ─── Pred 마스크 (3, Z, H, W) 불러오기 ───
    PRED_PATH  = Path(pred_root) / f"{stem}.nii.gz"
    pred_masks = sitk.GetArrayFromImage(sitk.ReadImage(str(PRED_PATH)))
    # 이전 채널 순서 [TC,WT,ET] → [WT,TC,ET]
    pred_masks = pred_masks[[1, 0, 2], ...]

    n_slices = raw_img.shape[0]  # 슬라이스 개수

    for z in range(n_slices):
        bg      = raw_img[z]           # (H, W) input image
        # GT masks
        wt_gt   = gt_masks[0, z]
        tc_gt   = gt_masks[1, z]
        et_gt   = gt_masks[2, z]
        # Pred masks
        wt_pr   = pred_masks[0, z]
        tc_pr   = pred_masks[1, z]
        et_pr   = pred_masks[2, z]

        # 1행 3열 subplot
        #fig, (ax_img, ax_gt, ax_pr) = plt.subplots(1, 3, figsize=(12, 4))
        fig, (ax_img, ax_gt, ax_pr) = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("black")
        fig.patch.set_alpha(1.0)

        # ─── Input Image ───
        ax_img.imshow(bg, cmap="gray", origin="lower")
        ax_img.set_title("Image"); ax_img.axis("off")

        # ─── GT overlay ───
        ax_gt.set_facecolor("black")
        ax_gt.axis("off")
        ax_gt.imshow(mask_to_rgba(wt_gt, (1, 0, 0, 0.5)), origin="lower")  # WT: red
        ax_gt.imshow(mask_to_rgba(tc_gt, (0, 1, 0, 0.5)), origin="lower")  # TC: green
        ax_gt.imshow(mask_to_rgba(et_gt, (1, 1, 0, 0.5)), origin="lower")  # ET: yellow
        ax_gt.set_title("GT")

        # ─── Ours overlay ───
        ax_pr.set_facecolor("black")
        ax_pr.axis("off")
        ax_pr.imshow(mask_to_rgba(wt_pr, (1, 0, 0, 0.5)), origin="lower")
        ax_pr.imshow(mask_to_rgba(tc_pr, (0, 1, 0, 0.5)), origin="lower")
        ax_pr.imshow(mask_to_rgba(et_pr, (1, 1, 0, 0.5)), origin="lower")
        ax_pr.set_title("Ours")

        plt.tight_layout()
        save_path = out_dir / f"{stem}_slice{z:03d}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✔ Saved {save_path}")

if __name__ == "__main__":
    main()
