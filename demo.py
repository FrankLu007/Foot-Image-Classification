import os
import csv
import torch, numpy
import torch, csv
import argparse
import torchvision.transforms
from PIL import Image
from model import ResNet
from tqdm import trange

if __name__ == '__main__' :

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--top", type = str, default = None)
    parser.add_argument("--left", type = str, default = None)
    parser.add_argument("--right", type = str, default = None)
    parser.add_argument("--bottom", type = str, default = None)
    parser.add_argument("--sv", type = str, default = 'baseline.csv')
    args = parser.parse_args()

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((int(args.sz * 400), int(args.sz * 120))),
        # torchvision.transforms.GaussianBlur(99, (0.1, 2)),
        # torchvision.transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
    ])

    model_left = torch.load(args.left)
    model_left.eval()
    model_top = torch.load(args.top)
    model_top.eval()
    model_right = torch.load(args.right)
    model_right.eval()
    model_bottom = torch.load(args.bottom)
    model_bottom.eval()
    torch.backends.cudnn.benchmark = True
    label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777]).cuda()

    outputs = [['images', 'x1', 'y1', 'x2', 'y2']]
    for index in trange(1, 1001):
        images = torch.stack([transform_train(Image.open('..\\Downloads\\test\\images\\image_%04d.png'%index)) for _ in range(args.bs)]).cuda()
        with torch.no_grad() :
            left = model_left(images[:, :, int(args.sz * 50) : int(args.sz * 200), :]).mean(0) + label_mean[0]
            top = model_top(images[:, :, int(args.sz * 50) : int(args.sz * 200), :]).mean(0) + label_mean[1]
            right = model_right(images[:, :, int(args.sz * 300) : int(args.sz * 400), :]).mean(0) + label_mean[2]
            bottom = model_bottom(images[:, :, int(args.sz * 300) : int(args.sz * 400), :]).mean(0) + label_mean[3]
        outputs.append(['image_%04d.png'%index] + left.tolist() + top.tolist() + right.tolist() + bottom.tolist())

        del top, bottom, images, left, right

    with open(args.sv, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(outputs)
