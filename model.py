import torch.nn
import torchvision.models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

<<<<<<< HEAD
class block(torch.nn.Module):
    def __init__(self, size) -> None:
        super(block, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(size, size), torch.nn.ReLU(inplace = True), torch.nn.Linear(size, size))
        self.out = torch.nn.ReLU(inplace = True)
    def forward(self, x):
        identity = x
        return self.out(identity + self.fc(x))
=======
class ResNet(torch.nn.Module):
    def __init__(self, OutputSize):
        super(ResNet, self).__init__()
        ImageNet = torchvision.models.resnet18(pretrained = True, progress = True)
        self.FeatureExtractor = torch.nn.Sequential(*list(ImageNet.children())[:5])
        self.layer1 = list(ImageNet.children())[5]
        self.layer2 = list(ImageNet.children())[6]
        self.layer3 = list(ImageNet.children())[7]
        sz = 64 + 128 + 256 + 512
        self.fc = torch.nn.Sequential(torch.nn.Linear(sz, sz), torch.nn.Linear(sz, OutputSize))
    def forward(self, x):
        a = self.FeatureExtractor(x)
        b = self.layer1(a)
        c = self.layer2(b)
        d = self.layer3(c)

        a = torch.flatten(F.adaptive_avg_pool2d(a, (1, 1)), 1)
        b = torch.flatten(F.adaptive_avg_pool2d(b, (1, 1)), 1)
        c = torch.flatten(F.adaptive_avg_pool2d(c, (1, 1)), 1)
        d = torch.flatten(F.adaptive_avg_pool2d(d, (1, 1)), 1)
        all = torch.cat((a, b, c, d), dim = 1)
        return self.fc(torch.flatten(all, 1))
>>>>>>> 693a863c2f84a03dc6f3a4991b8f8aac369d55ba

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self, OutputSize):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b7')
        size = self.ImageNet._fc.in_features
        self.ImageNet._fc = torch.nn.Linear(size, size)
        self.fc = torch.nn.Sequential(torch.nn.ReLU(inplace = True), block(size), block(size), torch.nn.Linear(size, OutputSize))
    def forward(self, x):
        return self.fc(self.ImageNet(x))
