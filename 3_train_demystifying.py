import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss

set_determinism(123)
import os
from thop import profile
import torch
import time

data_dir = "/media/NAS/nas_187/huiseung/fullres/train"
logdir = f"/media/NAS/nas_187/huiseung/logs/segmamba/Demistifying_stride_modify"

model_save_path = os.path.join(logdir, "model")
# augmentation = "nomirror"
augmentation = True

env = "DDP"
max_epoch = 1000
batch_size = 1
val_every = 2
num_gpus = 2
device = "cuda:0"
roi_size = [128, 128, 128]


def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.augmentation = augmentation
        # from model_segmamba.segmamba import SegMamba
        from token_priority.token_priority_segmamba_stride_modify import SegMamba
        # from model_assm_only.irv2_mamba_assm_only import SegMamba

        self.model = SegMamba(in_chans=4,
                              out_chans=4,
                              depths=[2, 2, 2, 2],
                              feat_size=[48, 96, 192, 384])

        self.train_memory_usage = []
        self.val_memory_usage = []
        self.batch_size_for_memory = batch_size  # 💡 per-image memory 계산용
        self.val_latencies = []  # validation latency 저장 리스트 추가

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.train_process = 18
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label = self.get_input(batch)

        pred = self.model(image)

        loss = self.cross(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        return loss

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])

        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])

        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        torch.cuda.reset_peak_memory_stats()  # 스텝 단위 리셋(유지)
        torch.cuda.synchronize()

        start_time = time.time()

        image, label = self.get_input(batch)

        output = self.model(image)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        torch.cuda.synchronize()  # 💡 again, 정확한 끝 시점 측정
        end_time = time.time()

        # ✅ 한 batch 처리에 걸린 시간 (초)
        elapsed_time = end_time - start_time
        self.val_latencies.append(elapsed_time / image.shape[0])  # batch_size로 나눠 per-image

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        # 💡 Validation memory 측정 - 추가
        self.val_memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

        print(f"dices is {tc, wt, et}")

        mean_dice = (tc + wt + et) / 3

        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")

        # ✅ validation 끝난 후 평균 latency 계산 - 추가
        avg_latency = np.mean(self.val_latencies)
        print(f"Average Inference Latency (seconds/image): {avg_latency:.6f}")
        self.val_latencies = []  # 💡 다음 epoch 대비 초기화

        # 💡 TM, IM 최종 출력 (per image 기준) - 추가
        if len(self.train_memory_usage) > 0:
            avg_tm = np.mean(self.train_memory_usage)
            print(f"Average Training Memory Usage (TM): {avg_tm:.0f} MB")

        if len(self.val_memory_usage) > 0:
            avg_im = np.mean(self.val_memory_usage)
            print(f"Average Inference Memory Usage (IM): {avg_im:.0f} MB")

    def measure_macs(self):
        device = self.local_rank if self.ddp else self.device  # DDP면 local_rank, 아니면 "cuda:0"
        self.model.to(device)
        self.model.eval()

        dummy_input = torch.randn(1, 4, 128, 128, 128, device=device)

        with torch.no_grad():
            macs, params = profile(self.model, inputs=(dummy_input,))

        print(f"Params: {params / 1e6:.2f} M, MACs: {macs / 1e9:.2f} GMACs")

if __name__ == "__main__":
    torch.set_num_threads(4)  # 💡 여기에 추가!
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17759,
                           training_script=__file__)
    trainer.measure_macs()  # 학습 시작 전에 MACs 계산
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
