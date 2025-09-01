import numpy as np

# 파일 경로
path_ours = "/media/NAS/nas_187/huiseung/prediction_results/ase_only_with_globaltoken/result_metrics/epoch1000_best_model.npy"
path_base = "/media/NAS/nas_187/huiseung/prediction_results/base_new/result_metrics/epoch1000_best_model.npy"

# 1) 불러오기
ours = np.load(path_ours, allow_pickle=False)   # shape = (250, 3, 2)
base = np.load(path_base, allow_pickle=False)

print("ours shape:", ours.shape)
print("base shape:", base.shape)

# 2) mean dice 만 추출 (마지막 축 index=0)
#    axis=1 이 클래스(WT,TC,ET), axis=2 는 [mean_dice, other_metric]
ours_dice = ours[:, :, 0]   # shape = (250, 3)
base_dice = base[:, :, 0]   # shape = (250, 3)

# 3) 차이 계산
diff_dice = ours_dice - base_dice  # shape = (250, 3)
print("diff_dice shape:", diff_dice.shape)

# 4-A) 클래스·샘플 단위 최대 개선 지점
flat_idx = diff_dice.argmax()       # flatten 된 인덱스
case_idx, class_idx = np.unravel_index(flat_idx, diff_dice.shape)
labels = ["WT", "TC", "ET"]
print(f"▶ 단일 클래스 최대 개선:")
print(f"  샘플 index = {case_idx}, 클래스 = {labels[class_idx]}")
print(f"    baseline dice = {base_dice[case_idx, class_idx]:.4f}")
print(f"        ours dice = {ours_dice[case_idx, class_idx]:.4f}")
print(f"          diff  = {diff_dice[case_idx, class_idx]:.4f}")


avg_diff_per_case = diff_dice.mean(axis=1)  # shape = (250,)
# 4-B) 샘플별 평균 개선량이 가장 큰 샘플
top10 = np.argsort(-avg_diff_per_case)[:20]

print("▶ 샘플별 평균 dice 개선 Top-20")
for rank, idx in enumerate(top10, 1):
    print(f"{rank}. 샘플 index = {idx} ({idx+1}번째 샘플)")
    print(f"   baseline avg dice = {base_dice[idx].mean():.4f}")
    print(f"       ours avg dice = {ours_dice[idx].mean():.4f}")
    print(f"         diff          = {avg_diff_per_case[idx]:.4f}\n")
