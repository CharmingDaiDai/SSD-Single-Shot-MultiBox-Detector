import torch
import config
import data
import model
from trainer_no_ck import Trainer
import argparse

def main():
    """主函数，用于组装和启动训练过程"""
    parser = argparse.ArgumentParser(description="训练参数可通过命令行传入")
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--momentum', type=float, default=None, help='动量')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--step_size', type=int, default=None, help='LR调度步长')
    parser.add_argument('--gamma', type=float, default=None, help='LR调度gamma')
    parser.add_argument('--num_classes', type=int, default=None, help='类别数')
    args = parser.parse_args()

    # 使用命令行参数或config中的默认值
    lr = args.lr if args.lr is not None else config.LEARNING_RATE
    momentum = args.momentum if args.momentum is not None else config.MOMENTUM
    weight_decay = args.weight_decay if args.weight_decay is not None else config.WEIGHT_DECAY
    step_size = args.step_size if args.step_size is not None else config.LR_SCHEDULER_STEP_SIZE
    gamma = args.gamma if args.gamma is not None else config.LR_SCHEDULER_GAMMA
    num_classes = args.num_classes if args.num_classes is not None else config.NUM_CLASSES

    print("Initializing DataLoaders...")
    train_loader, val_loader = data.get_dataloaders()

    print("Initializing Model...")
    ssd_model = model.create_model(num_classes=num_classes, pretrained=True)

    print("Initializing Optimizer and LR Scheduler...")
    params = [p for p in ssd_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=ssd_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader
    )

    trainer.fit()

if __name__ == '__main__':
    main()
