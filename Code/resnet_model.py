import torch
import torch.nn as nn
import torchvision.models as models


class BreastCancerResNet18(nn.Module):
    """
    Custom ResNet18 for breast cancer classification (benign/malignant).
    """

    def __init__(self, pretrained=True):
        super(BreastCancerResNet18, self).__init__()
        # Load ResNet18 with pre-trained weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze all layers to use ResNet as a feature extractor
        """for param in self.resnet.parameters():
            param.requires_grad = False"""

        # Modify the fully connected layer for binary classification (benign/malignant)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 4)

    def forward(self, x):
        """ Forward pass through the model """
        return self.resnet(x)
