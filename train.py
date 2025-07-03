import torch
import config
import data
import model
from trainer_no_ck import Trainer

def main():
    """主函数，用于组装和启动训练过程"""
    print("Initializing DataLoaders...")
    train_loader, val_loader = data.get_dataloaders()

    print("Initializing Model...")
    ssd_model = model.create_model(num_classes=config.NUM_CLASSES, pretrained=True)

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