import numpy as np
import os

# 저장된 결과 파일 경로
results_root = "/media/NAS/nas_187/huiseung/prediction_results/Demistifying_0416"#DEMISTIFYING"#assm_only"#base_new" #"/media/NAS/nas_187/huiseung/prediction_results/segmamba"
pred_name = "epoch1000_best_model"#"epoch100_tmp" #"epoch1000_final_model" #"epoch1000_best" #"epoch1000_final_model" #"epoch1000_best"  #"epoch100"# "segmamba"  # 여기 pred_name은 저장된 npy 파일 이름 (예: segmamba.npy)

# 파일 로드
result_path = os.path.join(results_root, "result_metrics", f"{pred_name}.npy")
results = np.load(result_path)  # shape: (250, 3, 2)

# 클래스 이름 설정 (BraTS 기준: TC, WT, ET)
class_names = ["TC", "WT", "ET"]

print(f"\n📊 Metrics Summary for '{pred_name}' ({results.shape[0]} cases):\n")

# 평균 및 표준편차 출력
for i, class_name in enumerate(class_names):
    dice_mean = results[:, i, 0].mean()
    dice_std = results[:, i, 0].std()
    hd95_mean = results[:, i, 1].mean()
    hd95_std = results[:, i, 1].std()

    print(f"🧠 {class_name}:")
    print(f"  - Dice  : {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"  - HD95  : {hd95_mean:.4f} ± {hd95_std:.4f}\n")

# 전체 평균 (mean of means)
mean_dice = results[:, :, 0].mean()
mean_hd95 = results[:, :, 1].mean()

print(f"✅ Overall:")
print(f"  - Mean Dice : {mean_dice:.4f}")
print(f"  - Mean HD95 : {mean_hd95:.4f}")
