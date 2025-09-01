#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────── 설정 ───────────
PRED_FILE = Path("/media/NAS/nas_187/huiseung/prediction_results/base_new/epoch1000_best_model/BraTS-GLI-00008-001.nii.gz")
OUT_DIR   = Path("/media/NAS/nas_187/huiseung/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────── 데이터 로드 & 전처리 ───────────
data = nib.load(PRED_FILE).get_fdata()
data = np.squeeze(data)                    # shape = (240, 155, 3, 240)

# “3”인 축을 채널 축으로 보고 collapse → (240,155,240)
if data.ndim == 4 and 3 in data.shape:
    ch_axis = data.shape.index(3)
    data = np.argmax(data, axis=ch_axis)

if data.ndim != 3:
    raise RuntimeError(f"3D 볼륨이 아닙니다: shape={data.shape}")

# ──── (240,155,240) → (240,240,155) 로 축 재배열 ────
data = np.swapaxes(data, 1, 2)

# ─────────── 중앙 슬라이스 선택 ───────────
z = data.shape[2] // 2
slice_mid = data[:, :, z]

# ─────────── 시각화 및 저장 ───────────
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(slice_mid.T, cmap="gray", origin="lower")
ax.axis("off")
out_path = OUT_DIR / f"{PRED_FILE.stem}_z{z:03d}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✔ Saved: {out_path}")
