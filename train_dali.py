import torch, numpy
import argparse
from csv import reader
from model import ResNet, EfficientNetWithFC
from random import shuffle
import nvidia.dali.fn as fn
from nvidia.dali.types import Constant, DALIDataType
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import matplotlib
import numpy
import matplotlib.pyplot as plt

def forward(DataLoader, model, truth, optimizer = None) :

    TotalLoss = 0

    for data in DataLoader:
        # initialize
        torch.cuda.empty_cache()
        if optimizer :
            optimizer.zero_grad()

        # forward
        outputs = model(data[0]['data'])
        loss = torch.sqrt(((outputs - truth[data[0]['label'].reshape(-1)]) ** 2).sum(1)).mean()
        TotalLoss += loss.item()

        # update
        if optimizer :
            loss.backward()
            optimizer.step()
        del loss, outputs

    DataLoader.reset()
    TotalLoss /= DataLoader.__len__()
    print('%5.3f'%TotalLoss)

    return TotalLoss

# +
if __name__ == '__main__':
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--dir", type = int, default = 0)
    parser.add_argument("--sz", type = float, default = 1)
    parser.add_argument("--sv", type = str, default = 'tmp.weight')
    parser.add_argument("--ld", type = str, default = None)
    args = parser.parse_args()
    
    with open('data.csv', 'r') as file:
        DataList = list(reader(file, delimiter = ','))
    
    TrainingData = []
    truth = torch.zeros((len(DataList), 2)).cuda()
    for index, data in enumerate(DataList[:-160]):
        TrainingData.append(data[0].replace('\\', '/'))
        truth[index][0] = float(data[args.dir * 2 + 1])
        truth[index][1] = float(data[args.dir * 2 + 2])
#         truth[index][2] = float(data[args.dir * 2 + 3]) - 55.0105
#         truth[index][3] = float(data[args.dir * 2 + 4]) - 378.1777

    ValidationData = []
    for index, data in enumerate(DataList[-160:]):
        ValidationData.append(data[0].replace('\\', '/'))
        truth[index + len(TrainingData)][0] = float(data[args.dir * 2 + 1])
        truth[index + len(TrainingData)][1] = float(data[args.dir * 2 + 2])

    if args.dir < 1:
        truth[:, 0] -= 57.9607
        truth[:, 1] -= 134.3954
    else :
        truth[:, 0] -= 55.0105
        truth[:, 1] -= 378.1777
    W, H = int(args.sz * 120), int(args.sz * 400)
    ImageBytes = W * H * 3 * 4

    TrainingPipe = Pipeline(batch_size = args.bs, num_threads = 4, device_id = 0)
    with TrainingPipe:
        files, labels = fn.file_reader(files = TrainingData, shuffle_after_epoch = True)
        images = fn.image_decoder(files, device = 'cpu', use_fast_idct = True)
        # images = fn.image_decoder_slice(files, anchor, (w, h), device = 'cpu', normalized_anchor = False, normalized_shape = False, use_fast_idct = True)
        labels = fn.cast(labels, device = 'cpu', dtype = DALIDataType.INT64)
        images = fn.resize(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, size = (H, W))
        images = fn.cast(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
        images = fn.gaussian_blur(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, sigma = fn.uniform(range = (0.1, 2)), window_size = 11)
        images = fn.color_twist(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, brightness = fn.uniform(range = (0.5, 1.5)), contrast = fn.uniform(range = (0.5, 2.5)), saturation = fn.uniform(range = (0.1, 2)))
        images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
        images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
        TrainingPipe.set_outputs(images, labels)
    TrainingLoader = DALIClassificationIterator(TrainingPipe, last_batch_policy = LastBatchPolicy.DROP, size = len(DataList))

    ValidationPipe = Pipeline(batch_size = 160, num_threads = 4, device_id = 0)
    with ValidationPipe:
        files, labels = fn.file_reader(files = ValidationData)
        images = fn.image_decoder(files, device = 'cpu', use_fast_idct = True)
        # images = fn.image_decoder_slice(files, anchor, (w, h), device = 'cpu', normalized_anchor = False, normalized_shape = False, use_fast_idct = True)
        labels = fn.cast(labels, device = 'cpu', dtype = DALIDataType.INT64)
        images = fn.resize(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, size = (H, W))
        images = fn.cast(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
        images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
        images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
        ValidationPipe.set_outputs(images, labels)
    ValidationLoader = DALIClassificationIterator(ValidationPipe, last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True, size = 160)

    if args.ld:
        model = torch.load(args.ld)
    else:
        model = EfficientNetWithFC(2).cuda()

    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    model.eval()
    with torch.no_grad():
        print('Validation', end = ' ')
        BestLoss = forward(ValidationLoader, model, truth[-160:])
    
    for epoch in range(args.ep):
        print('\nEpoch : %3d'%epoch)
        
        model.train()
        print('Training', end = ' ')
        forward(TrainingLoader, model, truth[:-160], optimizer)
        
        model.eval()
        with torch.no_grad():
            print('Validation', end = ' ')
            loss = forward(ValidationLoader, model, truth[-160:])
        
        if loss < BestLoss:
            BestLoss = loss
            torch.save(model, args.sv)

#     import xgboost as xgb
#     from sklearn.multioutput import MultiOutputRegressor
#     params = {
#         'booster': 'gbtree',
#         'objective': 'reg:squarederror',
#         'gamma' : 0.1,
#         'max_depth' : 5,
#         'min_child_weight' : 2,
#         'num_parallel_tree' : 100,
#         'subsample' : 0.9,
#         'colsample_bytree' : 0.1,
#         'alpha' : 0.1,
#         'nthread': 4,
#     }
#     model_top = torch.load('top.weight')
#     model_top.eval()
#     model_bottom = torch.load('bottom.weight')
#     model_bottom.eval()
#     for data in TrainingLoader:
#         with torch.no_grad():
#             featureUP = model_top.ImageNet(data[0]['data'][:, :, 50 : 200]).cpu().numpy()
#             featureBO = model_bottom.ImageNet(data[0]['data'][:, :, 300:]).cpu().numpy()
#             labelsT = data[0]['label'].reshape(-1)
#         del data[0]['data']
        
    # for data in ValidationLoader:
    #     with torch.no_grad():
    #         featureV = model.ImageNet(data[0]['data']).cpu().numpy()
    #         labelsV = data[0]['label'].reshape(-1) + len(DataList) - 160
    #     del data[0]['data']

    


#     DataList = []
#     for index in range(1, 1001):
#         DataList += ['../Downloads/test/images/image_%04d.png'%index] * args.bs

#     TestingPipe = Pipeline(batch_size = 1000, num_threads = 4, device_id = 0)
#     with TestingPipe:
#         files, labels = fn.file_reader(files = DataList)
#         images = fn.image_decoder(files, device = 'cpu', use_fast_idct = True)
#         # images = fn.resize(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, size = (H, W))
#         # images = fn.gaussian_blur(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, sigma = fn.uniform(range = (0.1, 2)), window_size = 11)
#         # images = fn.color_twist(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, brightness = fn.uniform(range = (0.5, 1.5)), contrast = fn.uniform(range = (0.5, 2.5)), saturation = fn.uniform(range = (0.1, 2)))
#         images = fn.cast(images.gpu(), device = 'gpu', bytes_per_sample_hint = ImageBytes, dtype = DALIDataType.FLOAT)
#         images = fn.normalize(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, mean = Constant(numpy.array([[[190.6380, 207.2640, 202.5720]]])), stddev = Constant(numpy.array([[[85.2720, 68.6970, 81.4215]]])))
#         images = fn.transpose(images, device = 'gpu', bytes_per_sample_hint = ImageBytes, perm = [2, 0, 1])
#         TestingPipe.set_outputs(images, labels)
#     TestingLoader = DALIClassificationIterator(TestingPipe, size = 1000)

#     myans = numpy.zeros((1000, 4))
#     for data in TestingLoader:
#         with torch.no_grad():
#             test_UP = model_top.ImageNet(data[0]['data'][:, :, 50 : 200]).cpu().numpy()
#             test_BO = model_bottom.ImageNet(data[0]['data'][:, :, 300:]).cpu().numpy()
#             myans[:, :2] = model_top(data[0]['data'][:, :, 50 : 200]).cpu().numpy()
#             myans[:, 2:] = model_bottom(data[0]['data'][:, :, 300:]).cpu().numpy()
#         del data[0]['data']

#     dtrain = xgb.DMatrix(featureUP, truth[labelsT][:, 0])
#     A = xgb.train(params, dtrain, 10)
#     dtest = xgb.DMatrix(test_UP)
#     a = A.predict(dtest)
#     del A, dtrain
#     print(numpy.abs(a - myans[:, 0]).mean())
    

#     dtrain = xgb.DMatrix(featureUP, truth[labelsT][:, 1])
#     B = xgb.train(params, dtrain, 10)
#     b = B.predict(dtest)
#     del B, dtrain, dtest
#     print(numpy.abs(b - myans[:, 1]).mean())
#     print(numpy.sqrt((a - myans[:, 0]) ** 2 + (b - myans[:, 1]) ** 2).mean())
#     dtrain = xgb.DMatrix(featureBO, truth[labelsT][:, 2])
#     C = xgb.train(params, dtrain, 10)
#     dtest = xgb.DMatrix(test_BO)
#     c = C.predict(dtest)
#     del C, dtrain
#     dtrain = xgb.DMatrix(featureBO, truth[labelsT][:, 3])
#     D = xgb.train(params, dtrain, 10)
#     d = D.predict(dtest)
#     del D, dtrain, dtest

#     print(numpy.abs(c - myans[:, 2]).mean())
#     print(numpy.abs(d - myans[:, 3]).mean())
#     print(numpy.sqrt((c - myans[:, 2]) ** 2 + (d - myans[:, 3]) ** 2).mean())
#     outputs = [['images', 'x1', 'y1', 'x2', 'y2']]
#     for i in range(1, 1001):
#         outputs.append(['image_%04d.png'%i] + [a[i - 1], b[i - 1], c[i - 1], d[i - 1]])

#     from csv import writer
#     with open('xgb.csv', 'w', newline = '') as file:
#         out = writer(file)
#         out.writerows(outputs)
# -


