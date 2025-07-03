import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from PIL import Image
from torchvision.transforms.v2 import ToImage, ConvertImageDtype
import os
import glob

from utils.viz import draw_bounding_boxes

# --- 配置参数 ---
# 模型权重路径
# 注意：请确保这里指向一个您已经训练和保存的.pth文件
CHECKPOINT_PATH = 'checkpoints/ssd300_voc_epoch_50.pth'
# 待批量预测的图片目录
IMAGE_DIR = 'images/'
# 输出图片的保存路径
OUTPUT_DIR = 'outputs'
# 置信度阈值
SCORE_THRESHOLD = 0.5
# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 类别数量
NUM_CLASSES = 21

def predict_batch():
    print(f"Using device: {DEVICE}")

    # --- 1. 加载模型 ---
    print("Loading model...")
    # 创建模型结构，与训练时完全一致
    # 重点：设置 weights=None 和 pretrained_backbone=False 来创建一个完全"冷"的模型
    # 这样可以防止在加载本地权重之前进行任何不必要的下载
    model = ssd300_vgg16(weights=None, pretrained_backbone=False, num_classes=NUM_CLASSES)
    
    # 对于使用VGG作为骨干的SSD300，需要手动设置一下分类头的参数
    # 否则，即使num_classes对了，内部的通道数也可能不匹配
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES
    )

    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}")
        print("Please make sure you have trained the model and the path is correct.")
        return

    model.to(DEVICE)
    model.eval() # 设置为评估模式

    # --- 2. 查找所有图片 ---
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return
    
    # 支持多种常见的图片格式
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    image_paths += glob.glob(os.path.join(IMAGE_DIR, '*.jpeg'))
    image_paths += glob.glob(os.path.join(IMAGE_DIR, '*.png'))

    if not image_paths:
        print(f"No images found in directory: {IMAGE_DIR}")
        return

    print(f"Found {len(image_paths)} images to predict in '{IMAGE_DIR}'.")
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 定义变换
    transform = torchvision.transforms.Compose([
        ToImage(),
        ConvertImageDtype(torch.float)
    ])

    # --- 3. 循环处理每张图片 ---
    for image_path in image_paths:
        print(f"  -> Processing: {os.path.basename(image_path)}")
        try:
            image_pil = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"     Could not open or process image {image_path}, skipping. Error: {e}")
            continue
            
        image_tensor = transform(image_pil).to(DEVICE)
        image_tensor = image_tensor.unsqueeze(0)

        # --- 执行推理 ---
        with torch.no_grad():
            predictions = model(image_tensor)
        
        prediction = predictions[0]

        # --- 可视化并保存结果 ---
        output_image = draw_bounding_boxes(
            image_pil,
            prediction['boxes'],
            prediction['labels'],
            prediction['scores'],
            score_threshold=SCORE_THRESHOLD
        )

        output_filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, f"pred_{output_filename}")
        output_image.save(output_path)

    print(f"\nBatch prediction finished. All results saved in '{OUTPUT_DIR}'.")
    
if __name__ == '__main__':
    predict_batch() 