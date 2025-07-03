import torch
from PIL import Image, ImageDraw, ImageFont

# VOC 类别名称，需要与训练时保持一致
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 为不同类别生成不同颜色的边界框
# 这里我们用一个简单的颜色列表，可以通过索引循环使用
COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
    '#000075', '#808080'
]

def draw_bounding_boxes(image, boxes, labels, scores, score_threshold=0.5):
    """
    在图片上绘制边界框、类别和置信度。

    Args:
        image (PIL.Image.Image): 原始图片。
        boxes (torch.Tensor): 边界框坐标，形状为 [N, 4]，格式为 (xmin, ymin, xmax, ymax)。
        labels (torch.Tensor): 类别索引，形状为 [N]。
        scores (torch.Tensor): 置信度，形状为 [N]。
        score_threshold (float): 置信度阈值，低于此值的框不会被绘制。

    Returns:
        PIL.Image.Image: 绘制了边界框的图片。
    """
    img_to_draw = image.copy()
    draw = ImageDraw.Draw(img_to_draw)
    
    try:
        # 使用一个常见的字体，如果系统中没有，会使用Pillow的默认字体
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # 过滤掉低置信度的结果
    high_confidence_indices = scores > score_threshold
    boxes_to_draw = boxes[high_confidence_indices]
    labels_to_draw = labels[high_confidence_indices]
    scores_to_draw = scores[high_confidence_indices]

    for i in range(len(boxes_to_draw)):
        box = boxes_to_draw[i].tolist()
        label_index = labels_to_draw[i].item()
        score = scores_to_draw[i].item()

        # 获取类别名称和颜色
        class_name = VOC_CLASSES[label_index]
        color = COLORS[label_index % len(COLORS)]

        # 绘制边界框
        draw.rectangle(box, outline=color, width=3)

        # 准备要绘制的文本
        text = f"{class_name}: {score:.2f}"
        
        # 获取文本框的大小
        try:
            # 新版 Pillow
            text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
        except AttributeError:
            # 旧版 Pillow
            text_w, text_h = draw.textsize(text, font=font)
            text_bbox = (box[0], box[1] - 15, box[0] + text_w, box[1] - 15 + text_h)

        # 绘制文本背景
        draw.rectangle(text_bbox, fill=color)
        
        # 绘制文本
        draw.text((box[0], text_bbox[1]), text, fill="black", font=font)

    return img_to_draw 