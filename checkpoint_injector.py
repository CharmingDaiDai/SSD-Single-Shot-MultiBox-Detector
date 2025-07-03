import threading
import os
import re
import torch
import time

from config import CHECKPOINT_DIR

CHECKPOINT_INTERVAL_SECONDS = 300  # 5分钟

def find_latest_checkpoint(directory):
    """查找最新的检查点文件（基于epoch次数）。"""
    os.makedirs(directory, exist_ok=True)
    files = [f for f in os.listdir(directory) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not files:
        return None
    epoch_pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    valid_files = [(int(match.group(1)), f) for f in files if (match := epoch_pattern.match(f))]
    if not valid_files:
        return None
    latest_file = max(valid_files, key=lambda item: item[0])
    return os.path.join(directory, latest_file[1])

def save_checkpoint_periodically(trainer, stop_event):
    """后台线程，定期保存检查点。"""
    while not stop_event.is_set():
        if stop_event.wait(timeout=CHECKPOINT_INTERVAL_SECONDS):
            break
        current_epoch = trainer.lr_scheduler.last_epoch
        if current_epoch == 0:
            continue
        state_to_save = {
            "epoch": current_epoch,
            "best_map": trainer.best_map,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.lr_scheduler.state_dict(),
        }
        filepath = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{current_epoch:04d}.pth")
        try:
            torch.save(state_to_save, filepath)
            print(f"\n[Checkpoint Saver] --> 成功保存检查点到 {filepath} <--", flush=True)
        except Exception as e:
            print(f"\n[Checkpoint Saver] --> 保存检查点失败: {e} <--")
    print("\n[Checkpoint Saver] 线程已停止。")

def add_checkpointing(BaseTrainer):
    """
    一个高阶函数（工厂），接收一个基础 Trainer 类，
    返回一个带有检查点加载和保存功能的新 Trainer 类。
    """
    class CheckpointedTrainerWrapper:
        def __init__(self, model, optimizer, lr_scheduler, *args, **kwargs):
            self.model = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            
            best_map_from_checkpoint = 0.0
            latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)

            if latest_checkpoint_path:
                print(f"发现最新的检查点: {latest_checkpoint_path}")
                choice = ''
                while choice not in ["y", "n"]:
                    choice = input("是否从该检查点恢复训练? (y/n): ").lower().strip()

                if choice == "y":
                    print("正在加载检查点...")
                    try:
                        checkpoint = torch.load(latest_checkpoint_path, map_location=self.model.device)
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        best_map_from_checkpoint = checkpoint.get("best_map", 0.0)
                        start_epoch = checkpoint.get("epoch", 0) + 1
                        # 此处不修改config.EPOCHS，让fit循环决定正确的范围
                        print(f"检查点加载成功，将从 Epoch {start_epoch} 继续训练。")
                    except Exception as e:
                        print(f"加载检查点失败: {e}。将开始新的训练。")
                else:
                    delete_choice = ''
                    while delete_choice not in ['y', 'n']:
                        delete_choice = input("您选择不恢复训练。是否要删除所有旧的检查点? (y/n): ").lower().strip()
                    if delete_choice == 'y':
                        print(f"正在删除目录 '{CHECKPOINT_DIR}' 中的所有检查点文件...")
                        try:
                            for filename in os.listdir(CHECKPOINT_DIR):
                                file_path = os.path.join(CHECKPOINT_DIR, filename)
                                if os.path.isfile(file_path): os.remove(file_path)
                            print("所有检查点已成功删除。")
                        except Exception as e:
                            print(f"删除检查点时出错: {e}")
            
            # 使用已经加载好状态的组件来实例化真正的Trainer
            self.trainer = BaseTrainer(model, optimizer, lr_scheduler, *args, **kwargs)
            self.trainer.best_map = best_map_from_checkpoint

        def fit(self):
            """启动后台保存线程并开始训练。"""
            stop_saver_event = threading.Event()
            saver_thread = threading.Thread(
                target=save_checkpoint_periodically,
                args=(self.trainer, stop_saver_event),
                daemon=True,
            )
            saver_thread.start()

            try:
                self.trainer.fit()
            except KeyboardInterrupt:
                print("\n捕获到用户中断 (Ctrl+C)，正在准备退出...")
            finally:
                print("正在停止检查点保存线程...")
                stop_saver_event.set()
                saver_thread.join(timeout=5)
                print("程序已安全退出。")

    return CheckpointedTrainerWrapper 