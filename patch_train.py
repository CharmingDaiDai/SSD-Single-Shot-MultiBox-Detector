import train
from trainer_no_ck import Trainer
import torch
import os
import re

# 保存原始方法
original_train_one_epoch = Trainer._train_one_epoch
original_fit = Trainer.fit

CHECKPOINT_DIR = "tmp"  # 检查点目录
SAVE_INTERVAL = 5  # 保存间隔
CKPT_KEYS = {
    "model": "model_state_dict",
    "optimizer": "optimizer_state_dict",
    "scheduler": "scheduler_state_dict",
    "epoch": "epoch",
    "best_map": "best_map"
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def find_latest_checkpoint():
    files = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]
    if not files:
        return None
    # 只保留能正确提取出数字的文件
    valid = []
    for f in files:
        m = re.match(r"checkpoint_epoch_(\d+)\.pth", f)
        if m:
            valid.append((int(m.group(1)), f))
    if not valid:
        return None
    valid.sort()
    return os.path.join(CHECKPOINT_DIR, valid[-1][1])


def patched_train_one_epoch(self, epoch):
    # 调用原始训练方法
    original_train_one_epoch(self, epoch)
    # 每 SAVE_INTERVAL 个 epoch 保存一次 checkpoint
    if epoch % SAVE_INTERVAL == 0:
        checkpoint = {
            CKPT_KEYS["model"]: self.model.state_dict(),
            CKPT_KEYS["optimizer"]: self.optimizer.state_dict(),
            CKPT_KEYS["scheduler"]: self.lr_scheduler.state_dict(),
            CKPT_KEYS["epoch"]: epoch,
            CKPT_KEYS["best_map"]: getattr(self, "best_map", 0.0),
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth")
        torch.save(checkpoint, ckpt_path)
        print(f"[MonkeyPatch] Checkpoint saved at {ckpt_path}")


def patched_fit(self):
    # 自动查找并恢复最新checkpoint
    latest_ckpt = find_latest_checkpoint()
    start_epoch = 1
    if latest_ckpt:
        print(f"[MonkeyPatch] Found checkpoint: {latest_ckpt}, loading...")
        ckpt = torch.load(latest_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt[CKPT_KEYS["model"]])
        self.optimizer.load_state_dict(ckpt[CKPT_KEYS["optimizer"]])
        self.lr_scheduler.load_state_dict(ckpt[CKPT_KEYS["scheduler"]])
        self.best_map = ckpt.get(CKPT_KEYS["best_map"], 0.0)
        start_epoch = ckpt.get(CKPT_KEYS["epoch"], 0) + 1
        print(f"[MonkeyPatch] Resume from epoch {start_epoch}")
    else:
        print("[MonkeyPatch] No checkpoint found, training from scratch.")

    print("Starting training process...")
    for epoch in range(start_epoch, self.epochs + 1):
        self._train_one_epoch(epoch)
        current_map = 0.0
        if self.val_loader:
            current_map = self._validate(epoch)
        if current_map > self.best_map:
            self.best_map = current_map
            best_model_path = os.path.join(self.models_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(
                f"New best model saved with mAP: {self.best_map:.4f} to {best_model_path}"
            )
        self.lr_scheduler.step()
    print("Finished training.")


# 替换 Trainer 的方法
Trainer._train_one_epoch = patched_train_one_epoch
Trainer.fit = patched_fit

# 启动训练
if __name__ == "__main__":
    train.main()
