import csv, cv2
import torch, numpy
from PIL import Image
import torchvision
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
from torchvision import	 transforms
from skimage import feature

transform_train = transforms.Compose([
	# transforms.Resize((512, 512)),
	# torchvision.transforms.GaussianBlur(7, (20, 20)),
	# transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)), #b = (0.75, 1.25), c = (0.5, 2), s = (1, 1.5), h = 0.05
	transforms.ToTensor(),
	# transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
])
recover = transforms.ToPILImage()



if __name__ == '__main__' :
	with open('data.csv', 'r') as file:
		data = list(csv.reader(file, delimiter = ','))

	# for d in range(2, 10):
	# 	image = numpy.array(Image.open(data[d][0]).convert('L').filter(ImageFilter.ModeFilter(7)).filter(ImageFilter.CONTOUR))
	# 	image[int(data[d][2]) - 5 : int(data[d][2]) + 5, int(data[d][1]) - 5 : int(data[d][1]) + 5] = 0
	# 	image = Image.fromarray(image)
	# 	image.show()
	# 	image = image.rotate(10, center = (int(data[d][1]), int(data[d][2])), fillcolor = 255)
	# 	image.show()
	# 	exit()
	s = torch.stack([transform_train(Image.open(d[0]).filter(ImageFilter.ModeFilter(7))) for d in data])
	print(s.mean((0, 2, 3)), s.std((0, 2, 3)))
	# tensor([0.7692, 0.8491, 0.8141]) tensor([0.3519, 0.2634, 0.3376])
	# tensor(0.8204) tensor(0.2309)
	# img = feature.canny(image.astype(float)[:, :, 0] / 255, sigma = 1)
	# blur_img = feature.canny(cv2.medianBlur(image[:, :, 0], 13).astype(float) / 255, sigma = 1)
