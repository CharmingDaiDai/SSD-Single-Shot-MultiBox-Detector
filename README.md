# SSD 物体检测项目

这是一个基于 PyTorch 实现的 SSD (Single Shot MultiBox Detector) 物体检测模型。项目代码使用 PASCAL VOC 2007 数据集进行训练，并实现了完整的训练、验证、预测和检查点管理流程。

## ✨ 功能特性

- **模型实现**: 基于 `torchvision` 的 `ssd300_vgg16` 模型。
- **数据处理**: 适配 PASCAL VOC 数据集，支持训练集和验证集加载。
- **模型训练**:
    - 使用 `tqdm` 提供美观、实时的训练进度条，动态显示损失值。
    - 自动保存验证集上 **mAP 最高** 的模型到 `models/best_model.pth`。
    - 定期保存训练检查点（checkpoint）到 `checkpoints/` 目录。
- **断点续训**:
    - 启动训练时，自动检测最新的检查点。
    - 用户可选择从检查点恢复训练，或开始一次全新的训练。
    - 若选择不恢复，会提示用户是否要清理所有旧的检查点。
- **批量预测**: 提供 `predict.py` 脚本，可对 `images/` 目录下的所有图片进行批量预测，并将结果保存在 `outputs/` 目录下。

## 🚀 环境准备

### 1. 数据集准备

本项目使用 PASCAL VOC 2007 数据集。请在项目根目录执行以下命令下载并解压数据集：

```bash
# 下载训练/验证集和测试集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 解压文件
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
```
执行完毕后，根目录下应出现 `VOCdevkit` 文件夹。

### 2. 环境与依赖安装

推荐使用 Python 3.12 版本。首先，安装所有必需的依赖库：

```bash
pip install -r requirements.txt
```

## 🔧 如何使用

### 训练模型

直接运行 `run_training.py` 脚本即可开始训练。

```bash
python run_training.py
```

- **恢复训练**: 如果 `checkpoints/` 目录下存在检查点，程序会提示您是否要恢复训练。输入 `y` 将从最新的检查点继续。
- **全新训练**: 如果选择不恢复训练（输入 `n`），程序会接着询问您是否要删除所有旧的检查点，方便您开始一次干净的训练。
- **最佳模型**: 训练过程中，mAP 最高的模型权重会被自动保存到 `models/best_model.pth`。

### 进行预测

1.  将您需要预测的图片放入 `images/` 文件夹。
2.  确保 `models/best_model.pth` 文件存在（通过上面的训练过程生成）。
3.  运行 `predict.py` 脚本。

```bash
python predict.py
```

脚本会自动加载最佳模型，对 `images/` 目录下的所有图片进行预测。带有标注框的可视化结果将保存在 `outputs/` 目录中。