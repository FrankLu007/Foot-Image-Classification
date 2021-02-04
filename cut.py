import csv
import random

directory = '..\\Downloads\\train\\'
with open(directory + 'annotation.csv') as file:
	data = list(csv.reader(file, delimiter = ','))[1:]
	for d in data:
		d[0] = directory + 'images\\' + d[0]

directory = '..\\Downloads\\train_20210106\\'
with open(directory + 'annotation.csv') as file:
	data += list(csv.reader(file, delimiter = ','))[1:]
	for d in data[1000:]:
		d[0] = directory + 'images\\' + d[0]

random.shuffle(data)

with open('data.csv', 'w', newline = '') as file:
	writer = csv.writer(file)
	writer.writerows(data)
