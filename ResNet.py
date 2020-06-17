import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride = 1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(out_planes)

		self.shortcut = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False),
			nn.BatchNorm2d(out_planes),
		)

	def forward(self, x):
		opt = F.relu(self.bn1(self.conv1(x)))
		opt = self.bn2(self.conv2(opt))
		opt += self.shortcut(x)
		opt = F.relu(opt)
		return opt

class ResNet(nn.Module):
	def __init__(self, block):
		super(ResNet, self).__init__()
		self.link_planes = 16
		self.conv1 = nn.Conv2d(3, 16, kernel_size = 7, stride = 2, padding = 5, bias = False)#kernel_size = 7, stride = 2, padding = 5
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self.layer_maker(block, 16, 5, 1)#96
		self.layer2 = self.layer_maker(block, 32, 6, 2)#48
		self.layer3 = self.layer_maker(block, 64, 8, 2)#24
		self.layer4 = self.layer_maker(block, 128, 5, 2)#12
		self.fc1 = nn.Linear(128*3*3, 32)
		self.fc2 = nn.Linear(32, 2)


	def layer_maker(self, block, planes, n_blocks, first_stride):
		strides = [first_stride] + [1]*(n_blocks - 1)
		layers = []
		for i_stride in strides:
			layers.append(block(self.link_planes, planes, i_stride))
			self.link_planes = planes
		return nn.Sequential(*layers)

	def forward(self, x):
		opt = F.relu(self.bn1(self.conv1(x)))
		opt = self.layer1(opt)
		opt = self.layer2(opt)
		opt = self.layer3(opt)
		opt = self.layer4(opt)
		opt = F.avg_pool2d(opt, 4)
		opt = opt.view(opt.size(0), -1)
		opt = F.relu(self.fc1(opt))
		f = opt
		opt = self.fc2(opt)
		return opt, f





