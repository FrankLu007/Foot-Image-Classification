import torch, numpy
import argparse
from csv import writer
from model import ResNet
import nvidia.dali.fn as fn
from nvidia.dali.types import Constant, DALIDataType
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

if __name__ == '__main__' :

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 100)
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--top", type = str, default = None)
    parser.add_argument("--bottom", type = str, default = None)
    parser.add_argument("--sv", type = str, default = 'baseline.csv')
    args = parser.parse_args()

    DataList = []
    labels = []
    Ids = [1]
    for index in range(1, 1001):
        DataList += ['../Downloads/test/images/image_%04d.png'%index] * args.bs
        labels += [index] * args.bs

    W, H = int(args.sz * 120), int(args.sz * 400)
    ImageBytes = W * H * 3 * 4

    TestingPipe = Pipeline(batch_size = args.bs, num_threads = 4, device_id = 0)
    with TestingPipe:
        files, labels = fn.file_reader(files = DataList, labels = labels)
        images = fn.image_decoder(files, device = 'cpu', use_fast_idct = True)
        images = fn.resize(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, size = (H, W))
        images = fn.gaussian_blur(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, sigma = fn.uniform(range = (0.1, 2)), window_size = 11)
        images = fn.color_twist(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, brightness = fn.uniform(range = (0.5, 1.5)), contrast = fn.uniform(range = (0.5, 2.5)), saturation = fn.uniform(range = (0.1, 2)))
        images = fn.cast(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
        images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
        images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
        TestingPipe.set_outputs(images, labels)
    TestingLoader = DALIClassificationIterator(TestingPipe, size = 1000 * args.bs)


    model_top = torch.load(args.top)
    model_top.eval()
    model_bottom = torch.load(args.bottom)
    model_bottom.eval()
    torch.backends.cudnn.benchmark = True
    label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777]).cuda()
    label_std = torch.Tensor([7.7420, 14.6187, 24.0051, 7.5642]).cuda()

    outputs = [['images', 'x1', 'y1', 'x2', 'y2']]
    index = 1
    for data in TestingLoader:

        with torch.no_grad() :
            top = model_top(data[0]['data']).mean(0) + label_mean[:2]
            bottom = model_bottom(data[0]['data']).mean(0) + label_mean[2:]
        outputs.append(['image_%04d.png'%index] + top.tolist() + bottom.tolist())

        del top, bottom
        index += 1

    with open(args.sv, 'w', newline = '') as file:
        out = writer(file)
        out.writerows(outputs)