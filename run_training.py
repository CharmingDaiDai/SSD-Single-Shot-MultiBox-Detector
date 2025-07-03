import threading
import os
import re
import torch

# --- 项目模块导入 ---
import config
import data
import model
from trainer_no_ck import Trainer

# --- 检查点逻辑 ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL_SECONDS = 30  # 30秒


def find_latest_checkpoint(directory):
    """查找最新的检查点文件（基于epoch次数）。"""
    os.makedirs(directory, exist_ok=True)
    files = [
        f
        for f in os.listdir(directory)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]
    if not files:
        return None
    # 正则表达式匹配 "checkpoint_epoch_数字.pth"
    epoch_pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    
    valid_files = []
    for f in files:
        match = epoch_pattern.match(f)
        if match:
            valid_files.append((int(match.group(1)), f))

    if not valid_files:
        return None

    # 找到epoch最大的文件
    latest_file = max(valid_files, key=lambda item: item[0])
    return os.path.join(directory, latest_file[1])


def save_checkpoint_periodically(trainer, stop_event):
    """后台线程，定期保存检查点。"""
    while not stop_event.is_set():
        # 等待一段时间或直到停止事件被设置
        if stop_event.wait(timeout=CHECKPOINT_INTERVAL_SECONDS):
            break # 如果事件被设置，则退出循环

        # 使用lr_scheduler的last_epoch作为当前迭代次数
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
            print(
                f"\n[Checkpoint Saver] --> 成功保存检查点到 {filepath} <--",
                flush=True,
            )
        except Exception as e:
            print(f"\n[Checkpoint Saver] --> 保存检查点失败: {e} <--")
            
    print("\n[Checkpoint Saver] 线程已停止。")


def main():
    """主程序入口"""
    # 1. 准备所有训练需要的组件
    print("Initializing DataLoaders...")
    train_loader, val_loader = data.get_dataloaders()

    print("Initializing Model...")
    ssd_model = model.create_model(num_classes=config.NUM_CLASSES, pretrained=True)
    ssd_model.to(config.DEVICE)

    print("Initializing Optimizer and LR Scheduler...")
    params = [p for p in ssd_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP_SIZE,
        gamma=config.LR_SCHEDULER_GAMMA
    )

    # 2. 检查是否有最新的检查点
    latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
    best_map_from_checkpoint = 0.0

    if latest_checkpoint_path:
        print(f"发现最新的检查点: {latest_checkpoint_path}")
        while True:
            choice = input("是否从该检查点恢复训练? (y/n): ").lower().strip()
            if choice in ["y", "n"]:
                break

        if choice == "y":
            print("正在加载检查点...")
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=config.DEVICE)
                ssd_model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                best_map_from_checkpoint = checkpoint.get("best_map", 0.0)
                
                start_epoch = checkpoint.get("epoch", 0) + 1
                config.EPOCHS = max(config.EPOCHS, start_epoch)
                
                print(f"检查点加载成功，将从 Epoch {start_epoch} 继续训练。")

            except Exception as e:
                print(f"加载检查点失败: {e}。将开始新的训练。")
        else:
            # 如果用户选择不恢复，则询问是否删除旧的检查点
            while True:
                delete_choice = input("您选择不恢复训练。是否要删除所有旧的检查点? (y/n): ").lower().strip()
                if delete_choice in ['y', 'n']:
                    break
            
            if delete_choice == 'y':
                print(f"正在删除目录 '{CHECKPOINT_DIR}' 中的所有检查点文件...")
                try:
                    for filename in os.listdir(CHECKPOINT_DIR):
                        file_path = os.path.join(CHECKPOINT_DIR, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f" - 已删除: {filename}")
                    print("所有检查点已成功删除。将开始新的训练。")
                except Exception as e:
                    print(f"删除检查点时出错: {e}")

    # 3. 实例化 Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=ssd_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader
    )
    trainer.best_map = best_map_from_checkpoint
    
    # 4. 准备并启动后台的检查点保存线程
    stop_saver_event = threading.Event()
    saver_thread = threading.Thread(
        target=save_checkpoint_periodically,
        args=(trainer, stop_saver_event),
        daemon=True,
    )
    saver_thread.start()

    # 5. 在主线程中开始训练
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n捕获到用户中断 (Ctrl+C)，正在准备退出...")
    finally:
        # 6. 训练结束或中断后，通知后台线程停止
        print("正在停止检查点保存线程...")
        stop_saver_event.set()
        saver_thread.join(timeout=5) # 等待线程结束
        print("程序已安全退出。")


if __name__ == "__main__":
    main()
