import argparse
import torch, csv
import torchvision
from PIL import Image
from model import EfficientNetWithFC, ResNet

class FootDataset(torch.utils.data.Dataset):
	def __init__(self, DataList, transform):
		super(FootDataset, self).__init__()
		self.DataList = DataList
		self.transform = transform
		self.label_mean = torch.Tensor([57.9607, 134.3954, 55.0105, 378.1777])
	def __len__(self):
		return len(self.DataList)
	def __getitem__(self, index):
		data = self.DataList[index]
		image = self.transform(Image.open(data[0]))
		label = (torch.Tensor([float(value) for value in data[1:]]) - self.label_mean)

		return image, label

def forward(DataLoader, model, optimizer = None) :

    TotalLoss = 0

    for inputs, labels in DataLoader:

        # initialize
        torch.cuda.empty_cache()
        if optimizer :
            optimizer.zero_grad()

        # forward
        inputs = inputs.cuda()
        labels = labels.reshape(-1, 2).cuda()
        outputs = model(inputs).reshape(-1, 2)
        del inputs
        loss = torch.sqrt(((outputs - labels) ** 2).sum(1)).mean()
        TotalLoss += loss.item()

        # update
        if optimizer :
            loss.backward()
            optimizer.step()
            scaler.update()
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
	parser.add_argument("--sv", type = str, default = 'tmp.weight')
	parser.add_argument("--ld", type = str, default = None)
	args = parser.parse_args()
    
	with open('data.csv', 'r') as file:
		data = list(csv.reader(file, delimiter = ','))

	transform_train = torchvision.transforms.Compose([
		torchvision.transforms.Resize((int(args.sz * 400), int(args.sz * 120))),
		torchvision.transforms.GaussianBlur(99, (0.1, 2)),
		torchvision.transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
	])

	transform_test = torchvision.transforms.Compose([
		torchvision.transforms.Resize((int(args.sz * 400), int(args.sz * 120))),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
	])

	TrainingSet = FootDataset(data[:-150], transform_train)
	ValidationSet = FootDataset(data[-150:], transform_test)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args.bs, pin_memory = True, drop_last = True, shuffle = True, num_workers = 0)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = args.bs, num_workers = 0)

	if args.ld:
		model = torch.load(args.ld)
	else:
		model = ResNet(4).cuda()
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