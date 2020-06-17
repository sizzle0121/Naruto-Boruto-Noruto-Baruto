import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
import torchvision.utils as utils
import os
import numpy as np
from PIL import Image

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


transform_ = transforms.Compose([
	#transforms.CenterCrop(size = (200, 200)),
	transforms.Resize(size = (192, 192), interpolation = 1),
])

transform_flip = transforms.Compose([
	#transforms.CenterCrop(size = (200, 200)),
	transforms.Resize(size = (192, 192), interpolation = 1),
	transforms.RandomHorizontalFlip(p = 1),
])

transform_crop = transforms.Compose([
	#transforms.CenterCrop(size = (200, 200)),
	transforms.Resize(size = (192, 192), interpolation = 1),
	transforms.RandomCrop(192, padding = 7),
])

transform_flip_crop = transforms.Compose([
	#transforms.CenterCrop(size = (200, 200)),
	transforms.Resize(size = (192, 192), interpolation = 1),
	transforms.RandomCrop(192, padding = 7),
	transforms.RandomHorizontalFlip(p = 1),
])



def pil2np(img):
	arr = np.array(img)
	arr = arr.transpose(2, 0, 1)
	return arr.astype(np.float32)/255


class Train_Dataset(Data.Dataset):
	def __init__(self):
		comic = 'naruto'
		label = 0
		self.Images = []
		for i in range(2):
			if i == 1:
				comic = 'boruto'
				label = 1
			self.files = os.listdir('./Train_set/'+comic+'/')
			for f in self.files:
				img = Image.open('./Train_set/'+comic+'/'+f)
				img1 = pil2np(transform_(img))
				img2 = pil2np(transform_flip(img))
				#img3 = pil2np(transform_crop(img))
				#img4 = pil2np(transform_flip_crop(img))
				self.Images.append([img1, np.array([ label ])])
				self.Images.append([img2, np.array([ label ])])
				#self.Images.append([img3, np.array([ label ])])
				#self.Images.append([img4, np.array([ label ])])

	def __getitem__(self, idx):
		X = FloatTensor(self.Images[idx][0])
		Y = LongTensor(self.Images[idx][1])
		return X, Y

	def __len__(self):
		return len(self.Images)



class Test_Dataset(Data.Dataset):
	def __init__(self):
		comic = 'naruto'
		label = 0
		self.Images = []
		for i in range(2):
			if i == 1:
				comic = 'boruto'
				label = 1
			self.files = os.listdir('./Test_set/'+comic+'/')
			for f in self.files:
				img = Image.open('./Test_set/'+comic+'/'+f)
				img1 = pil2np(transform_(img))
				self.Images.append([img1, np.array([ label ])])

	def __getitem__(self, idx):
		X = FloatTensor(self.Images[idx][0])
		Y = LongTensor(self.Images[idx][1])
		return X, Y

	def __len__(self):
		return len(self.Images)





