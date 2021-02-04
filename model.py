import torch.nn
import torchvision.models
from efficientnet_pytorch import EfficientNet

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.ImageNet = torchvision.models.resnet18(pretrained = True, progress = True)
        self.fc = torch.nn.Linear(1000, 4)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b7')
        self.fc = torch.nn.Linear(1000, 4)
    def forward(self, x):
        return self.fc(self.ImageNet(x))