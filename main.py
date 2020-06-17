import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms

from ResNet import *
from preprocess import *
from VallinaCNN import *
from argparse import ArgumentParser
import numpy as np
import csv
import os


LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


#TEST = False
#LR = 0.01#0.000005
#EPOCH = 200
#BATCH_SIZE = 32
#MODELS = 'ResNet'


parser = ArgumentParser()
parser.add_argument('-m', '--model', help = 'select models', dest = 'models', default = 'ResNet')
parser.add_argument('-t', '--test_mode', help = 'use testing mode', dest = 'test', default = 'on')
parser.add_argument('-lr', '--learning_rate', help = 'assign learning rate', dest = 'lr', default = 0.01)
parser.add_argument('-e', '--epoch', help = 'number of epoch to run', dest = 'epoch', default = 200)
parser.add_argument('-l', '--loader', help = 'select loader', dest = 'load', default = 'test')
parser.add_argument('-en', '--encode', help = 'encode features', dest = 'encode', default = 'off')
args = parser.parse_args()

MODELS = args.models
TEST = False if args.test == 'off' else True
LR = args.lr
EPOCH = int(args.epoch)
BATCH_SIZE = 1
ENCODE = True if args.encode == 'on' else False


if MODELS == 'ResNet' or MODELS == 'ResNet_DT' or MODELS == 'ResNet_SVM':
	model = ResNet(BasicBlock).cuda()
elif MODELS == 'VallinaCNN':
	model = VallinaCNN(CNNBlock).cuda()
#optimizer = optim.Adam(model.parameters(), lr = LR)
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9, weight_decay = 0.0001)
loss_func = nn.CrossEntropyLoss().cuda()

if TEST == False:
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in')
elif TEST == True:
	if MODELS == 'ResNet' or MODELS == 'ResNet_DT' or MODELS == 'ResNet_SVM':
		model.load_state_dict(torch.load('ResNet.pth'))
		if MODELS == 'ResNet_DT' or MODELS == 'ResNet_SVM':
			from DecisionTree import *
	elif MODELS == 'VallinaCNN':
		model.load_state_dict(torch.load('VallinaCNN.pth'))
	model.eval()



print('Test Mode: ', TEST)
print('Model: '+MODELS)
print('Learning Rate: %f' %(LR))
print('Epoch: %d' %(EPOCH))


print('\nBuilding Dataset...')
Train_set = Train_Dataset()
Test_set = Test_Dataset()


#print('Training Data Size: %d' %(len(Train_set)))
#print('Testing Data Size: %d' %(len(Test_set)))


Train_Loader = Data.DataLoader(dataset = Train_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 1)
Test_Loader = Data.DataLoader(dataset = Test_set, batch_size = 32, shuffle = False, num_workers = 1)
Loader = Train_Loader if args.load == 'train' else Test_Loader
print('Building Completed')


def adjust_lr(optimizer, epoch):
	lr = LR
	if epoch >= 40 and epoch < 70:
		lr /= 10
	elif epoch >= 70:
		lr /= 100

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


print('\nStart Running')
best_acc = 0
if TEST == False:
	train_log = csv.writer(open('train_log.csv', 'w'))
	test_log = csv.writer(open('test_log.csv', 'w'))


Features = []
for epoch in range(EPOCH):
	if TEST == False:
		train_loss = 0
		train_total = 0
		train_correct = 0
		adjust_lr(optimizer, epoch)
		for batch_idx, (X, Y) in enumerate(Test_Loader):
			X = Variable(X).cuda()
			Y = Variable(Y).cuda()
			opt, _ = model(X)

			loss = loss_func(opt, Y.squeeze())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			_, predict = torch.max(opt, 1)
			predict = predict.view(-1, 1)
			train_correct += predict.eq(Y.data).cpu().numpy().sum()
			train_total += Y.size(0)
			train_loss += loss.data.cpu().item()
			del loss
		print('Epoch: %d\t| Loss: %.4f\t| Accuracy: %.4f' %(epoch, train_loss, train_correct/train_total))
		train_log.writerow([train_loss])

	test_loss = 0
	test_total = 0
	test_correct = 0
	test_cnt = 0

	for batch_idx, (X, Y) in enumerate(Loader):
		X = Variable(X).cuda()
		Y = Variable(Y).cuda()
		opt, features = model(X)
		test_total += Y.size(0)
		features = features.data.cpu().numpy()
		if MODELS == 'ResNet' or MODELS == 'VallinaCNN':
			if ENCODE:
				features = features.reshape(-1, 1)
				Features.append([features, Y.data.cpu().numpy()])
			_, predict = torch.max(opt, 1)
			predict = predict.view(-1, 1)
			test_correct += predict.eq(Y.data).cpu().numpy().sum()
		elif MODELS == 'ResNet_DT':
			features = features.reshape(-1, 32)
			predict = tree.predict(features)
			#print(predict)
			target = Y.squeeze().data.cpu().numpy()
			#print(target)
			test_correct += np.equal(predict, target).sum()

		elif MODELS == 'ResNet_SVM':
			features = features.reshape(-1, 32)
			predict = SVM.predict(features)
			target = Y.squeeze().data.cpu().numpy()
			test_correct += np.equal(predict, target).sum()

	acc = test_correct/test_total
	print('[Testing] Epoch: %d\t| Accuracy: %.4f' %(epoch, acc))
	#print('Correct: %d, Total: %d' %(test_correct, test_total))
	#print()

	if TEST == False:
		test_log.writerow([acc])
		if acc >= best_acc:
			if MODELS == 'ResNet':
				torch.save(model.state_dict(), 'ResNet.pth')
			elif MODELS == 'VallinaCNN':
				torch.save(model.state_dict(), 'VallinaCNN.pth')
			best_acc = acc
	if ENCODE:
		np.save('FEATURES_v3', Features)


