import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        return self.model(x)["out"]
