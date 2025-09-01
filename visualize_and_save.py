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
    bool_mask = mask.astype(bool)             # boolean 인덱싱용
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
        (labels == 1) | (labels == 2) | (labels == 3),  # WT (Whole Tumor: labels 1,2,3)
        (labels == 1) | (labels == 3),                   # TC (Tumor Core: labels 1 & 3)
        (labels == 3)                                   # ET (Enhancing Tumor: label 3)
    ]
    return torch.cat(result, dim=0).float()           # (3, Z, H, W)

def main():
    stem         = "BraTS-GLI-00136-000"
    raw_data_dir = "/media/NAS/nas_187/huiseung/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    pred_root    = "/media/NAS/nas_187/huiseung/prediction_results/base_new/epoch1000_best_model"
    out_dir      = Path("/media/NAS/nas_187/huiseung/visualizations/compare_raw_align") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── GT 원본 (Z, H, W) 불러오기 ───
    GT_PATH = Path(raw_data_dir) / stem / "seg.nii.gz"
    raw_gt  = sitk.GetArrayFromImage(sitk.ReadImage(str(GT_PATH))).astype(np.int32)

    # ─── GT 바이너리 마스크 (3, Z, H, W) 생성 ───
    gt_masks = convert_labels(torch.from_numpy(raw_gt)).numpy()

    # ─── Pred 마스크 (3, Z, H, W) 불러오기 ───
    PRED_PATH  = Path(pred_root) / f"{stem}.nii.gz"
    pred_masks = sitk.GetArrayFromImage(sitk.ReadImage(str(PRED_PATH)))
    # **여기서 Pred 채널 순서를 [WT,TC,ET]로 재배열합니다**
    pred_masks = pred_masks[[1, 0, 2], ...]  # 이전엔 [TC,WT,ET] → 이제 [WT,TC,ET]
    print("GT mask sums ▶ WT, TC, ET =",
          gt_masks[0].sum(), gt_masks[1].sum(), gt_masks[2].sum())
    print("Pred mask sums ▶ WT, TC, ET =",
          pred_masks[0].sum(), pred_masks[1].sum(), pred_masks[2].sum())

    # ─── 슬라이스별 시각화 및 저장 ───
    n_slices = raw_gt.shape[0]  # 예: 155
    for z in range(n_slices):
        bg      = raw_gt[z]           # (H, W) 그레이스케일 배경
        bg      = raw_gt[z]           # (H, W) 그레이스케일 배경
        wt_gt   = gt_masks[0, z]      # GT Whole Tumor mask
        tc_gt   = gt_masks[1, z]      # GT Tumor Core mask
        et_gt   = gt_masks[2, z]      # GT Enhancing Tumor mask
        wt_pred = pred_masks[0, z]    # Pred Whole Tumor mask
        tc_pred = pred_masks[1, z]    # Pred Tumor Core mask
        et_pred = pred_masks[2, z]    # Pred Enhancing Tumor mask

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # GT overlays:
        #  - 빨간색 (R=1): WT (Whole Tumor)
        axes[0,0].imshow(bg, cmap="gray", origin="lower")
        axes[0,0].imshow(mask_to_rgba(wt_gt,  (1, 0, 0, 0.5)), origin="lower")
        axes[0,0].set_title("GT - WT"); axes[0,0].axis("off")
        #  - 초록색 (G=1): TC (Tumor Core)
        axes[0,1].imshow(bg, cmap="gray", origin="lower")
        axes[0,1].imshow(mask_to_rgba(tc_gt,  (0, 1, 0, 0.5)), origin="lower")
        axes[0,1].set_title("GT - TC"); axes[0,1].axis("off")
        #  - 파란색 (B=1): ET (Enhancing Tumor)
        axes[0,2].imshow(bg, cmap="gray", origin="lower")
        axes[0,2].imshow(mask_to_rgba(et_gt,  (0, 0, 1, 0.5)), origin="lower")
        axes[0,2].set_title("GT - ET"); axes[0,2].axis("off")

        # Pred overlays:
        #  - 빨간색 (R=1): WT (Whole Tumor)
        axes[1,0].imshow(bg, cmap="gray", origin="lower")
        axes[1,0].imshow(mask_to_rgba(wt_pred,(1, 0, 0, 0.5)), origin="lower")
        axes[1,0].set_title("Pred - WT"); axes[1,0].axis("off")
        #  - 초록색 (G=1): TC (Tumor Core)
        axes[1,1].imshow(bg, cmap="gray", origin="lower")
        axes[1,1].imshow(mask_to_rgba(tc_pred,(0, 1, 0, 0.5)), origin="lower")
        axes[1,1].set_title("Pred - TC"); axes[1,1].axis("off")
        #  - 파란색 (B=1): ET (Enhancing Tumor)
        axes[1,2].imshow(bg, cmap="gray", origin="lower")
        axes[1,2].imshow(mask_to_rgba(et_pred,(0, 0, 1, 0.5)), origin="lower")
        axes[1,2].set_title("Pred - ET"); axes[1,2].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"{stem}_slice{z:03d}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✔ Saved {save_path}")

if __name__ == "__main__":
    main()
