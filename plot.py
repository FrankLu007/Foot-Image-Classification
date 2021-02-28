# +
import csv, cv2
import torch
from PIL import Image, ImageOps
import torchvision
from torchvision import	 transforms
# %matplotlib inline

import matplotlib
import numpy
import matplotlib.pyplot as plt
# -

transform_train = transforms.Compose([
	# transforms.Resize((512, 512)),
	# torchvision.transforms.GaussianBlur(99, (0.1, 2)),
	# transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.1, 2), contrast = (0.5, 2.5)), #b = (0.75, 1.25), c = (0.5, 2), s = (1, 1.5), h = 0.05
    transforms.ToTensor(),
    # transforms.Normalize((0.7476, 0.8128, 0.7944), (0.3344, 0.2694, 0.3193)),
])

# +
if __name__ == '__main__' :
    with open('data.csv', 'r') as file:
        data = list(csv.reader(file, delimiter = ','))
    from random import shuffle
    shuffle(data)
    print(data[0])
    image = numpy.array(Image.open(data[0][0].replace('\\', '/')))
    
#     image = numpy.array(ImageOps.grayscale(Image.open(data[0][0].replace('\\', '/'))))
    a, b, c, d = int(data[0][1]), int(data[0][2]), int(data[0][3]), int(data[0][4])
#     image[b - 10 : b + 10, a - 10 : a + 10] = 0
    plt.imshow(image)
    plt.show()
    
    from skimage import feature
    for i in range(3):
        edges1 = feature.canny(image.astype(float)[:, :, i] / 255)
        edges2 = feature.canny(image.astype(float)[:, :, i] / 255, sigma = 5.5)

        plt.imshow(edges2)
        plt.show()
        edges2[b - 10 : b + 10, a - 10 : a + 10] = 1
        edges2[d - 10 : d + 10, c - 10 : c + 10] = 1
        plt.imshow(edges2)
        plt.show()
# -




