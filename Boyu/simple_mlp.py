import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim

class SimpleMLP(nn.Module):

	def __init__(self,
				in_size,
				hidden_size,
				out_size = 2, 
				dropout = 0.1):
		super(SimpleMLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(in_size, hidden_size)
		self.relu = nn.ReLU()
		self.w2 = nn.Linear(hidden_size, out_size)
		self.dropout = nn.Dropout(p = dropout)
		self.sigmoid = nn.Sigmoid()
		self.initialize()

	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		# print(self.w1, self.w2)

	def forward(self, x):

		h1 = self.w1(x)
		a = self.relu(h1)
		# a = self.dropout(a)
		h2 = self.w2(a)
		# h2 = self.sigmoid(h2)
		return h2

def load_data_MLP(path):

	with open(path, 'rb') as f:
		(low_list, not_low_list) = pickle.load(f)

	# [number of users, number of features]
	ls = np.stack([user[1] for user in low_list])
	nls = np.stack([user[1] for user in not_low_list])
	print('low shape: [{}], not low shape: [{}]'.format(ls.shape, nls.shape))
	assert ls.shape[1] == nls.shape[1]
	in_size = ls.shape[1]

	# add label [number of users, number of features]
	ls_label = np.stack([np.asarray([1, 0]) for user in ls])
	ls = np.concatenate((ls, ls_label), axis = 1)
	nls_label = np.stack([np.asarray([0, 1]) for user in nls])
	nls = np.concatenate((nls, nls_label), axis = 1)
	# print('ls example: [{}], nls example: [{}]'.format(ls[0], nls[0]))

	# [number of users, number of features + number of labels]
	data = np.concatenate((ls, nls), axis = 0)
	np.random.shuffle(data)
	data = torch.from_numpy(data).float()
	return data, in_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, in_size = load_data_MLP(
		path = 'lambda_vectors_60.pkl')

# X: [batch, num of features]
# Y: [batch, num of labels]
X, Y = data[:, :in_size].to(device), data[:, in_size:].to(device)
print('X: {}, Y: {}'.format(X.shape, Y.shape))

mlp = SimpleMLP(
		in_size = in_size, 
		hidden_size = 200).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mlp.parameters())

loss_history = []
best_loss = float('inf')
n_epoch = 1500
for epoch in range(n_epoch):

	optimizer.zero_grad()
	outputs = mlp(X)
	loss = criterion(outputs, Y)
	loss.backward()
	optimizer.step()

	if epoch % 10 == 0:
		print('Epoch [{}/{}] Loss: {}'.format(epoch, n_epoch, loss.item()))
	loss_history.append(loss.item())

# store the best hidden representations
with open('./best_representation.pkl', 'wb') as f:
	with torch.no_grad():
		h = mlp.w1(X)
		a = mlp.relu(h)
		best_representation = a.numpy()

	# put into 2 groups
	ls = []
	nls = []
	for idx, user in enumerate(Y):
		vec = best_representation[idx, :]
		# ls
		if user[0] == 1:
			ls.append(vec)
		# nls
		else:
			nls.append(vec)

	ls = np.stack(ls)
	nls = np.stack(nls)
	print(ls.shape, nls.shape)
	pickle.dump((ls, nls), f)
