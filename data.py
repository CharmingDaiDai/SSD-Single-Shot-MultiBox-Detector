import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCDetection
from torchvision.transforms.v2 import Compose, ToImage, ConvertImageDtype, RandomHorizontalFlip

import config

def get_transform(train):
    """获取数据预处理/增强的变换"""
    transforms = []
    transforms.append(ToImage())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        transforms.append(RandomHorizontalFlip(p=0.5))
    return Compose(transforms)

def collate_fn(batch):
    """自定义的collate_fn来处理DataLoader中的数据"""
    return tuple(zip(*batch))

def get_dataloaders():
    """
    创建并返回训练和验证的 DataLoader。
    """
    dataset_train_full = VOCDetection(
        root=config.DATA_DIR,
        year='2007',
        image_set='trainval',
        download=False,
        transforms=get_transform(train=True)
    )

    dataset_val_full = VOCDetection(
        root=config.DATA_DIR,
        year='2007',
        image_set='test',
        download=False,
        transforms=get_transform(train=False)
    )
    
    if config.USE_SUBSET:
        print(f"Using a subset of the dataset: {config.TRAIN_SUBSET_SIZE} for training and {config.VAL_SUBSET_SIZE} for validation.")
        train_indices = list(range(config.TRAIN_SUBSET_SIZE))
        val_indices = list(range(config.VAL_SUBSET_SIZE))
        
        dataset_train = Subset(dataset_train_full, train_indices)
        dataset_val = Subset(dataset_val_full, val_indices)
    else:
        dataset_train = dataset_train_full
        dataset_val = dataset_val_full

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader 