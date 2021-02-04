import csv
import torch
from PIL import Image
import torchvision
from torchvision import	 transforms

transform_train = transforms.Compose([
	# transforms.Resize((512, 512)),
	# torchvision.transforms.GaussianBlur(99, (0.1, 2)),
	# transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)), #b = (0.75, 1.25), c = (0.5, 2), s = (1, 1.5), h = 0.05
    transforms.ToTensor(),
    # transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
])

if __name__ == '__main__' :
	with open('data.csv', 'r') as file:
		data = list(csv.reader(file, delimiter = ','))

	x1, y1, x2, y2 = [], [], [], []
	for d in data:
		x1.append(int(d[1]))
		y1.append(int(d[2]))
		x2.append(int(d[3]))
		y2.append(int(d[4]))

	x1 = torch.Tensor(x1)
	y1 = torch.Tensor(y1)
	x2 = torch.Tensor(x2)
	y2 = torch.Tensor(y2)

	print(x1.mean(), x1.std())
	print(y1.mean(), y1.std())
	print(x2.mean(), x2.std())
	print(y2.mean(), y2.std())


