import os
import csv
import torch, numpy
import torch, csv
import argparse
import torchvision.transforms
from PIL import Image
from model import ResNet

if __name__ == '__main__' :

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--ld", type = str, default = None)
    args = parser.parse_args()

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.sz * 400, args.sz * 120)),
        # torchvision.transforms.GaussianBlur(99, (0.1, 2)),
        # torchvision.transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
    ])

    model = torch.load(args.ld)
    model.eval()
    torch.backends.cudnn.benchmark = True
    label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777]).cuda()
    label_std = torch.Tensor([7.7420, 14.6187, 24.0051, 7.5642]).cuda()

    outputs = [['images', 'x1', 'y1', 'x2', 'y2']]
    for index in range(1, 1001):
        image = transform_train(Image.open('..\\Downloads\\test\\images\\image_%04d.png'%index)).reshape(1, 3, args.sz * 400, args.sz * 120).cuda()
        with torch.no_grad():
            output = model(image).reshape(-1) + label_mean
        outputs.append(['image_%04d.png'%index] + output.tolist())

        del output, image

    with open('baseline.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(outputs)
