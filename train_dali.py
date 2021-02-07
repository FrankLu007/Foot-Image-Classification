import torch, numpy
import argparse
from csv import reader
from model import ResNet
from random import shuffle
import nvidia.dali.fn as fn
from nvidia.dali.types import Constant, DALIDataType
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

def forward(DataLoader, model, optimizer = None) :

    TotalLoss = 0

    for data in DataLoader:

        # initialize
        torch.cuda.empty_cache()
        if optimizer :
            optimizer.zero_grad()

        # forward
        labels = data[0]['label'].cuda()
        outputs = model(data[0]['data']).reshape(-1)
        loss = torch.abs(outputs - labels).mean()
        TotalLoss += loss.item()

        # update
        if optimizer :
            loss.backward()
            optimizer.step()
        del loss, labels, outputs

    TotalLoss /= DataLoader.__len__()
    print('%5.3f'%TotalLoss)

    return TotalLoss

if __name__ == '__main__':
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--dir", type = int, default = 0)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    args = parser.parse_args()
    
    with open('data.csv', 'r') as file:
        DataList = list(reader(file, delimiter = ','))
    
    TrainingData = [[], []]
    for data in DataList[:-150]:
        TrainingData[0].append(data[0].replace('\\', '/'))
        TrainingData[1].append(float(data[args.dir + 1]))
    ValidationData = [[], []]
    for data in DataList[-150:]:
        ValidationData[0].append(data[0].replace('\\', '/'))
        ValidationData[1].append(float(data[args.dir + 1]))

    if args.dir <= 1:
        anchor = (0, 50)
        w, h = 120, 150
    else :
        anchor = (0, 300)
        w, h = 120, 100
    W, H = int(args.sz * w), int(args.sz * h)
    ImageBytes = W * H * 3 * 4
        
    TrainingPipe = Pipeline(batch_size = args.bs, num_threads = 4, device_id = 0)
    with TrainingPipe:
        files, labels = fn.file_reader(files = TrainingData[0], labels = TrainingData[1], shuffle_after_epoch = True)
        images = fn.image_decoder_slice(files, anchor, (w, h), device = 'cpu', normalized_anchor = False, normalized_shape = False, use_fast_idct = True)
        images = fn.resize(images.gpu(), device = 'gpu', size = (H, W))
        images = fn.gaussian_blur(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, sigma = fn.uniform(range = (0.1, 2)), window_size = 11)
        images = fn.color_twist(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, brightness = fn.uniform(range = (0.5, 1.5)), contrast = fn.uniform(range = (0.5, 2.5)), saturation = fn.uniform(range = (0.1, 2)))
        images = fn.cast(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
        images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
        images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
        TrainingPipe.set_outputs(images, labels)
    TrainingLoader = DALIClassificationIterator(TrainingPipe, last_batch_policy = LastBatchPolicy.DROP)
    
    ValidationPipe = Pipeline(batch_size = args.bs, num_threads = 4, device_id = 0)
    with ValidationPipe:
        files, labels = fn.file_reader(files = ValidationData[0], labels = ValidationData[1])
        images = fn.image_decoder_slice(files, anchor, (w, h), device = 'cpu', normalized_anchor = False, normalized_shape = False, use_fast_idct = True)
        images = fn.resize(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, size = (H, W))
        images = fn.cast(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
        images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
        images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
        ValidationPipe.set_outputs(images, labels)
    ValidationLoader = DALIClassificationIterator(ValidationPipe, last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True)

    if args.ld:
        model = torch.load(args.ld)
    else:
        model = ResNet(1).cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    model.eval()
    with torch.no_grad():
        print('Validation', end = ' ')
        BestLoss = forward(ValidationLoader, model)
    
    for epoch in range(args.ep):
        print('\nEpoch : %3d'%epoch)
        
        model.train()
        print('Training', end = ' ')
        forward(TrainingLoader, model, optimizer, scaler)
        
        model.eval()
        with torch.no_grad():
            print('Validation', end = ' ')
            loss = forward(ValidationLoader, model)
        
        if loss < BestLoss:
            BestLoss = loss
            torch.save(model, args.sv)


