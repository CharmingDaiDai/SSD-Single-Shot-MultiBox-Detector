import torch

# --- 基础配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 21  # 20 个 VOC 类别 + 1 个背景

# --- 目录配置 ---
DATA_DIR = '.'  # 指向 VOCdevkit 的父目录
CHECKPOINT_DIR = 'checkpoints' # 直接在根目录创建
MODELS_DIR = 'models' # 直接在根目录创建
OUTPUT_DIR = 'outputs' # 直接在根目录创建

# --- 训练配置 ---
BATCH_SIZE = 4
LEARNING_RATE = 0.002
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 60
LR_SCHEDULER_STEP_SIZE = 20
LR_SCHEDULER_GAMMA = 0.9

# --- 数据集子集配置 ---
USE_SUBSET = True 
TRAIN_SUBSET_SIZE = 8
VAL_SUBSET_SIZE = 8

# --- VOC 类别 ---
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
] 