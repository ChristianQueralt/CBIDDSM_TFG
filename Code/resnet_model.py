import torch
import torch.nn as nn
import torchvision.models as models

class BreastCancerResNet18(nn.Module):
    """
    Custom ResNet18 for breast cancer classification (4 classes).
    """

    def __init__(self, pretrained=True):
        super(BreastCancerResNet18, self).__init__()

        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze entire backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Re-enable training only for the new classification head
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # 4-class output
        )

        # Ensure new fc layers are trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
