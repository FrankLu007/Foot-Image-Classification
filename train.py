import argparse, numpy
import torch, csv, cv2
import torchvision
from PIL import Image, ImageChops
from model import ResNet, EfficientNetWithFC
from skimage import feature

class FootDataset(torch.utils.data.Dataset):
	def __init__(self, DataList, transform, direction, mode = None):
		super(FootDataset, self).__init__()
		self.DataList = DataList
		self.transform = transform
		self.direction = direction
		self.mode = mode
		if direction == 0:
			self.label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777])[:2]
		else:
			self.label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777])[2:]
	def __len__(self):
		return len(self.DataList)
	def __getitem__(self, index):
		data = self.DataList[index]
		image = Image.open(data[0].replace('\\', '/'))
		# image = cv2.filter2D(numpy.array(Image.open(data[0]), dtype = numpy.single)[300 : 400] / 255, -1, kernel = kernel)
		# image = numpy.array(Image.open(data[0]), dtype = float) / 255
		# blur_img = cv2.GaussianBlur(img, (0, 0), 100)
		# image = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

		# image[:, :, 0] = feature.canny(image[:, :, 0], sigma = 4)
		# image[:, :, 1] = feature.canny(image[:, :, 1], sigma = 4)
		# image[:, :, 2] = feature.canny(image[:, :, 2], sigma = 4)
		# image = Image.fromarray(image.astype('uint8') * 255)
		label = [int(value) for value in data[self.direction * 2 + 1 : self.direction * 2 + 3]]
		if self.mode:
			# x_limit, y_limit = min(120 - label[0], label[0]), min(label[1] - self.y_min, self.y_max - label[1])
			x_offset, y_offset = torch.randint(low = - 10,  high = 10, size = (2,))
			if torch.rand(size = (1, )) >= 0.5:
				image = image.transpose(Image.FLIP_LEFT_RIGHT)
				label[0] = 120 - label[0]
			image = numpy.array(ImageChops.offset(image, x_offset, y_offset))
			if x_offset > 0:
				image[:, :x_offset] = 255
			else:
				image[:, 120 + x_offset :] = 255
			if y_offset > 0:
				image[:y_offset] = 255
			else:
				image[400 + y_offset :] = 255
			image = Image.fromarray(image)
			label[0] += x_offset
			label[1] += y_offset
		return self.transform(image), torch.Tensor(label) - self.label_mean

def forward(DataLoader, model, optimizer = None) :

    TotalLoss = 0

    for inputs, labels in DataLoader:
        # initialize
        torch.cuda.empty_cache()
        if optimizer :
            optimizer.zero_grad()

        # forward
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        del inputs
        loss = torch.sqrt(((outputs - labels) ** 2).sum(1)).mean()
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
	parser.add_argument("--lr", type = float, default = 0.001)
	parser.add_argument("--bs", type = int, default = 16)
	parser.add_argument("--sz", type = float, default = 1)
	parser.add_argument("--dir", type = int, default = 0)
	parser.add_argument("--sv", type = str, default = 'tmp.weight')
	parser.add_argument("--ld", type = str, default = None)
	args = parser.parse_args()
    
	with open('data.csv', 'r') as file:
		data = list(csv.reader(file, delimiter = ','))

	transform_train = torchvision.transforms.Compose([
		torchvision.transforms.GaussianBlur(7, (0.1, 2)),
		torchvision.transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
	])

	transform_test = torchvision.transforms.Compose([
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
	])

	TrainingSet = FootDataset(data[:-160], transform_train, args.dir, mode = 'train')
	ValidationSet = FootDataset(data[-160:], transform_test, args.dir)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args.bs, pin_memory = True, drop_last = True, shuffle = True, num_workers = 0)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = args.bs, num_workers = 0)

	if args.ld:
		model = torch.load(args.ld)
	else:
		model = EfficientNetWithFC(2).cuda()
	torch.backends.cudnn.benchmark = True
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

	model.eval()
	with torch.no_grad() :
		print('Validatoin', end = ' ')
		BestLoss = forward(ValidationLoader, model)

	for epoch in range(args.ep) :

		print('\nEpoch : ' + str(epoch))

		model.train()
		print('Training', end = ' ')
		forward(TrainingLoader, model, optimizer)

		model.eval()
		with torch.no_grad() :
			print('Validatoin', end = ' ')
			loss = forward(ValidationLoader, model)

		if loss < BestLoss:
			BestLoss = loss
			torch.save(model, args.sv)