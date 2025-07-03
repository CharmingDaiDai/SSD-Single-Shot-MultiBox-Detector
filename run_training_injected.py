import torch

# --- 项目模块导入 ---
import config
import data
import model
from trainer_no_ck import Trainer
from checkpoint_injector import add_checkpointing


def main():
    """
    主程序入口：使用"注入"方式为 Trainer 添加检查点功能。
    """
    # 1. 使用注入器增强原始的 Trainer 类
    # add_checkpointing 函数返回一个包装了原始 Trainer 的新类
    CheckpointedTrainer = add_checkpointing(Trainer)

    # 2. 准备所有训练需要的组件
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
    
    # 3. 实例化增强后的 Trainer
    # 它的 __init__ 方法现在已经包含了所有检查点加载和用户交互逻辑
    print("Initializing Trainer...")
    trainer = CheckpointedTrainer(
        model=ssd_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 4. 开始训练
    # 它的 fit 方法现在已经包含了后台保存线程的管理逻辑
    print("Starting training via injected wrapper...")
    trainer.fit()
    print("Training finished.")


if __name__ == "__main__":
    main() 