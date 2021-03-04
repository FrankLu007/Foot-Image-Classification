import torch.nn
import torchvision.models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class block(torch.nn.Module):
    def __init__(self, size) -> None:
        super(block, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(size, size), torch.nn.ReLU(inplace = True), torch.nn.Linear(size, size))
        self.out = torch.nn.ReLU(inplace = True)
    def forward(self, x):
        identity = x
        return self.out(identity + self.fc(x))

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self, OutputSize):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b7')
        size = self.ImageNet._fc.in_features
        self.ImageNet._fc = torch.nn.Linear(size, size)
        self.fc = torch.nn.Sequential(torch.nn.ReLU(inplace = True), block(size), block(size), torch.nn.Linear(size, OutputSize))
    def forward(self, x):
        return self.fc(self.ImageNet(x))
