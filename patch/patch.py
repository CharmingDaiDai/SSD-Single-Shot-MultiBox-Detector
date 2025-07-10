import sys
import os
import re
import importlib
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import *

# 1. 先解析patch.py自己的参数
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="tmp", help="checkpoint保存目录")
parser.add_argument("--save_interval", type=int, default=5, help="保存间隔")
# 只解析已知参数，其余留给train.py
args, remaining = parser.parse_known_args()

CHECKPOINT_DIR = args.ckpt_dir
SAVE_INTERVAL = args.save_interval
CKPT_KEYS = {
    "model": "model_state_dict",
    "optimizer": "optimizer_state_dict",
    "scheduler": "scheduler_state_dict",
    "epoch": "epoch",
    "best_map": "best_map",
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 2. 把剩余参数重新写回sys.argv，供train.py使用
sys.argv = [sys.argv[0]] + remaining

# 3. 导入train模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
module = importlib.import_module("train")

# 需要知道的内容：
# - Trainer 类
# - 类的初始化方法
# - 模型的属性名称
# - 优化器的属性名称
# - 学习率调度器的属性名称
# - 训练的轮数
# - 如果保存最佳模型，还需要知道最佳 mAP 的属性名称
# - Trainer._train_one_epoch 方法
# - Trainer.fit 方法

# 保存原始方法
Trainer = module.Trainer
original_train_one_epoch = Trainer._train_one_epoch
original_fit = Trainer.fit


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
    module.main()
# python patch/patch.py --ckpt_dir myckpt --save_interval 5 --lr 0.002 --momentum 0.9
