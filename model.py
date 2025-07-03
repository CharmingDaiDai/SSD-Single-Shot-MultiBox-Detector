import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

import config

def create_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    创建 SSD300 VGG16 模型。
    """
    weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
    
    model = ssd300_vgg16(weights=weights)

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]

    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model 