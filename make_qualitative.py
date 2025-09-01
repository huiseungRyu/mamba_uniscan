#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

# ─── mask_to_rgba 정의 ───
def mask_to_rgba(mask: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    """
    2D mask 배열(mask.shape == (H, W))을 RGBA 이미지로 변환합니다.
    """
    bool_mask = mask.astype(bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[bool_mask] = (*color[:3], alpha)
    return rgba

# ─── convert_labels 정의 ───
def convert_labels(labels: torch.Tensor) -> np.ndarray:
    """
    GT 라벨맵 (Z, H, W)을 3채널 바이너리 마스크로 변환합니다:
      [0]=WT, [1]=TC, [2]=ET
    """
    labels = labels.unsqueeze(0)  # (1, Z, H, W)
    masks = torch.cat([
        (labels == 1) | (labels == 2) | (labels == 3),  # WT
        (labels == 1) | (labels == 3),                  # TC
        (labels == 3),                                  # ET
    ], dim=0).float()  # (3, Z, H, W)
    return masks.numpy()

# ─── NIfTI 불러오기 ───
def load_nifti_np(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def main():
    # --- 경로 설정 --- #
    raw_data_dir  = Path(
        "/media/NAS/nas_187/huiseung/BraTS2023/"
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )
    pred_base_dir = Path(
        "/media/NAS/nas_187/huiseung/"
        "prediction_results/base_new/epoch1000_best_model"
    )
    pred_ours_dir = Path(
        "/media/NAS/nas_187/huiseung/"
        "prediction_results/ase_only_with_globaltoken/epoch1000_best_model"
    )
    out_dir = Path(
        "/media/NAS/nas_187/huiseung/visualizations/qualitative_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Top3 케이스 --- #
    cases = [
        ("BraTS-GLI-00469-000", 77),
        ("BraTS-GLI-00360-000", 106),
        ("BraTS-GLI-01090-000", 71),
    ]

    # WT, TC, ET 색상 (R, G, B)
    colors = [
        (1.0, 0.0, 0.0),  # WT: red
        (0.0, 1.0, 0.0),  # TC: green
        (1.0, 1.0, 0.0),  # ET: yellow
    ]

    # 3행×4열 Figure 생성
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 4,
                             figsize=(16, 4 * n_cases),
                             constrained_layout=True)

    for i, (stem, z) in enumerate(cases):
        # Input (t1c)
        img_np = load_nifti_np(raw_data_dir / stem / "t1c.nii.gz").astype(np.float32)
        bg = img_np[z]

        # GT
        gt_np = load_nifti_np(raw_data_dir / stem / "seg.nii.gz").astype(np.int32)
        gt_masks = convert_labels(torch.from_numpy(gt_np))

        # Baseline 예측
        base_np = load_nifti_np(pred_base_dir / f"{stem}.nii.gz")
        base_masks = base_np[[1,0,2], ...]

        # Ours 예측
        ours_np = load_nifti_np(pred_ours_dir / f"{stem}.nii.gz")
        ours_masks = ours_np[[1,0,2], ...]

        # (0) Input - 회색 영상
        ax = axes[i, 0]
        ax.imshow(bg, cmap="gray", origin="lower")
        ax.set_title("Input", color="black")  # 제목을 Input만
        ax.axis("off")

        # (1) GT mask only - 검정 배경
        ax = axes[i, 1]
        black_bg = np.zeros_like(bg)
        ax.imshow(black_bg, cmap="gray", vmin=0, vmax=1, origin="lower")
        for c, col in enumerate(colors):
            ax.imshow(mask_to_rgba(gt_masks[c, z], col, alpha=0.5), origin="lower")
        ax.set_title("GT", color="black")
        ax.axis("off")

        # (2) Baseline mask only - 검정 배경
        ax = axes[i, 3]
        ax.imshow(black_bg, cmap="gray", vmin=0, vmax=1, origin="lower")
        for c, col in enumerate(colors):
            ax.imshow(mask_to_rgba(base_masks[c, z], col, alpha=0.5), origin="lower")
        ax.set_title("SegMamba", color="black")
        ax.axis("off")

        # (3) Ours mask only - 검정 배경
        ax = axes[i, 2]
        ax.imshow(black_bg, cmap="gray", vmin=0, vmax=1, origin="lower")
        for c, col in enumerate(colors):
            ax.imshow(mask_to_rgba(ours_masks[c, z], col, alpha=0.5), origin="lower")
        ax.set_title("Ours", color="black")
        ax.axis("off")

    # 저장 및 종료
    save_path = out_dir / "qualitative_top3.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)
    print(f"✔ Saved qualitative results to {save_path}")

if __name__ == "__main__":
    main()
