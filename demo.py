import csv, cv2
import torch, numpy
import argparse
import torchvision.transforms
from PIL import Image, ImageChops
from model import ResNet
from tqdm import trange
from skimage import feature

if __name__ == '__main__' :

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--sv", type = str, default = 'baseline.csv')
    parser.add_argument("--top", type = str, default = None)
    parser.add_argument("--bottom", type = str, default = None)
    args = parser.parse_args()

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.GaussianBlur(7, (0.1, 2)),
        torchvision.transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
    ])


    def preprocess(image, o, flip = False):
        # image = numpy.array(image, dtype = float) / 255
        # image[:, :, 0] = feature.canny(image[:, :, 0], sigma = 4)
        # image[:, :, 1] = feature.canny(image[:, :, 1], sigma = 4)
        # image[:, :, 2] = feature.canny(image[:, :, 2], sigma = 4)
        # image = Image.fromarray(image.astype('uint8') * 255)
        x_offset, y_offset = o[0], o[1]
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = numpy.array(ImageChops.offset(image, x_offset, y_offset))
        if x_offset > 0:
            image[:, :x_offset] = 255
        else:
            image[:, 120 + x_offset :] = 255
        if y_offset > 0:
            image[:y_offset] = 255
        else:
            image[400 + y_offset :] = 255
        if torch.rand(size = (1, )) >= 0.5:
                image[200:300] = 255
        return transform_train(Image.fromarray(image))

    top_model = torch.load(args.top)
    bottom_model = torch.load(args.bottom)
    torch.backends.cudnn.benchmark = True
    top_model.eval()
    bottom_model.eval()
    label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777]).cuda()

    outputs = [['images', 'x1', 'y1', 'x2', 'y2']]
    offset = torch.randint(low = -10, high = 10, size = (args.bs, 2)).cuda()
    for index in trange(1, 1001):
        image = Image.open('../Downloads/test/images/image_%04d.png'%index)
        inputs_flip = torch.stack([preprocess(image, o, True) for o in offset]).cuda()
        inputs = torch.stack([preprocess(image, o) for o in offset]).cuda()
        with torch.no_grad() :
            torch.cuda.empty_cache()
            top = (top_model(inputs) - offset).mean(0) + label_mean[:2]
            bottom = (bottom_model(inputs) - offset).mean(0) + label_mean[2:]
            top_flit = (top_model(inputs_flip) - offset).mean(0) + label_mean[:2]
            top_flit[0] = 120 - top_flit[0]
            bottom_flip = (bottom_model(inputs_flip) - offset).mean(0) + label_mean[2:]
            bottom_flip[0] = 120 - bottom_flip[0]
            top = (top + top_flit) / 2
            bottom = (bottom + bottom_flip) / 2

        # inputs[:, :, int(top[1]) - 5 : int(top[1]) + 5, int(top[0]) - 5 : int(top[0]) + 5] = 1
        # inputs[:, :, int(bottom[1]) - 5 : int(bottom[1]) + 5, int(bottom[0]) - 5 : int(bottom[0]) + 5] = 1
        outputs.append(['image_%04d.png'%index] + top.tolist() + bottom.tolist())

        del image, inputs, top, bottom, top_flit, bottom_flip

    with open(args.sv, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(outputs)
