import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim
import sys
sys.path.insert(0,'../zipped_files_from_www')
from helper import *

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
		return h2

# load the data for pretraining
# ONLY use the scale parameter when the data is the raw lambda file!!!
def load_data_MLP(path, scale):

	with open(path, 'rb') as f:
		(low_list, not_low_list) = pickle.load(f)

	# [number of users, number of features]
	ls = np.stack([user[1] * scale for user in low_list])
	nls = np.stack([user[1] * scale for user in not_low_list])
	print('low shape: [{}], not low shape: [{}]'.format(ls.shape, nls.shape))
	assert ls.shape[1] == nls.shape[1]
	in_size = ls.shape[1]
	ls_num = ls.shape[0]
	nls_num = nls.shape[0]

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
	return data, in_size, ls_num, nls_num


def load_data_MLP_with_userid(path, scale):

    with open(path, 'rb') as f:
        (low_list, not_low_list) = pickle.load(f)
    # [number of users, number of features]
    ls = np.stack([user[1][27:] * scale for user in low_list])
    nls = np.stack([user[1][27:] * scale for user in not_low_list])

    #print(ls)

    ls_users = np.stack([[user[0]] for user in low_list])
    nls_users = np.stack([[user[0]] for user in not_low_list])
    #print('low shape: [{}], not low shape: [{}]'.format(ls.shape, nls.shape))
    assert ls.shape[1] == nls.shape[1]
    in_size = ls.shape[1]
    ls_num = ls.shape[0]
    nls_num = nls.shape[0]

    # add label [number of users, number of features]
    ls_label = np.stack([np.asarray([1, 0]) for user in ls])
    ls = np.concatenate((ls_users,ls, ls_label), axis = 1)
    nls_label = np.stack([np.asarray([0, 1]) for user in nls])
    nls = np.concatenate((nls_users,nls, nls_label), axis = 1)
    # print('ls example: [{}], nls example: [{}]'.format(ls[0], nls[0]))

    # [number of users, number of features + number of labels]
    data = np.concatenate((ls, nls), axis = 0)
    np.random.shuffle(data)
    #data = torch.from_numpy(data).float()
    return data, in_size, ls_num, nls_num


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, in_size, ls_num, nls_num= load_data_MLP(
# 		path = 'lambda_vectors_with_user_ID.pkl', 
# 		scale = 10**5)


data, in_size, ls_num, nls_num= load_data_MLP_with_userid(
		path = 'compound_vectors_self_esteem.pkl', 
		scale = 1)

# data, in_size, ls_num, nls_num= load_data_MLP_with_userid(
# 		path = 'compound_vectors_psi.pkl', 
# 		scale = 1)


userid,X, Y = data[:, 0], data[:, 1:in_size+1], data[:, in_size+1:]
vfloat = np.vectorize(float)
X,Y = torch.from_numpy(vfloat(X)).float().to(device), torch.from_numpy(vfloat(Y)).float().to(device)
print('X: {}, Y: {}'.format(X.shape, Y.shape))
#assert(False)

# X: [batch, num of features]
# Y: [batch, num of labels]
# X, Y = data[:, :in_size].to(device), data[:, in_size:].to(device)
# print('X: {}, Y: {}'.format(X.shape, Y.shape))


ls_compound, nls_compound = load_pickle("compound_vectors_self_esteem.pkl")
psi_compound, npsi_compound = load_pickle('compound_vectors_psi.pkl')
psi_users = [i[0] for i in psi_compound]
npsi_users = [i[0] for i in npsi_compound]
ls_users = [i[0] for i in ls_compound]
nls_users = [i[0] for i in nls_compound]

mlp = SimpleMLP(
		in_size = in_size, 
		hidden_size = 10).to(device)

criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(mlp.parameters(), weight_decay = 0.0001)
optimizer = optim.SGD(mlp.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.00001)

loss_history = []
best_loss = float('inf')

n_epoch = 5000

for epoch in range(n_epoch):

	optimizer.zero_grad()
	outputs = mlp(X)
	loss = criterion(outputs, Y)
	loss.backward()
	optimizer.step()

	if epoch % 100 == 0:
		print('Epoch [{}/{}] Loss: {}'.format(epoch, n_epoch, loss.item()))
	loss_history.append(loss.item())



# final training acc.
with torch.no_grad():
	outputs = mlp(X)
	results = []
	for user in outputs:
		if user[0] >= user[1]:
			results.append(np.asarray([1, 0]))
		else:
			results.append(np.asarray([0, 1]))
	results = np.stack(results)
	Y_temp = Y.numpy() 

	correct = 0
	c_ls = 0
	c_nls = 0
	for idx, user in enumerate(results):
		if np.array_equal(Y_temp[idx], user):
			correct += 1
			if np.array_equal(np.asarray([1, 0]), user):
				c_ls += 1
			else:
				c_nls += 1
		#print('acc: {}, ls acc: {}, nls acc: {}'.format(correct / len(results), c_ls / ls_num, c_nls / nls_num))
	print('acc: {}, psi acc: {}, npsi acc: {}'.format(correct / len(results), c_ls / ls_num, c_nls / nls_num))





# store the best hidden representations
with open('./best_representation.pkl', 'wb') as f:
	with torch.no_grad():
		h = mlp.w1(X)
		a = mlp.relu(h)
		best_representation = a.numpy()
		best_lin_rep = h.numpy()

	# put into 2 groups
	ls = []
	nls = []
	psi=[]
	npsi = []
	ls_lin, nls_lin,psi_lin,npsi_lin = [],[],[],[]
	# for idx, user in enumerate(Y):
	# 	vec = best_representation[idx, :]
	# 	# ls
	# 	if user[0] == 1:
	# 		ls.append(vec)
	# 	# nls
	# 	else:
	# 		nls.append(vec)

	# ls = np.stack(ls)
	# nls = np.stack(nls)
	# print(ls.shape, nls.shape)
	# pickle.dump((ls, nls), f)


	for idx, user in enumerate(userid):
		vec = best_representation[idx, :]
		vec1 = best_lin_rep[idx,:]
		# ls
		if user in psi_users:
			psi.append((user,vec))
			psi_lin.append((user,vec1))
		# nls
		else:
			npsi.append((user,vec))
			npsi_lin.append((user,vec1))


		if user in ls_users:
			ls.append((user,vec))
			ls_lin.append((user,vec1))
		else:
			nls.append((user,vec))
			nls_lin.append((user,vec1))
	# print('\n\n\n\n\n')
	# print(psi[0])
	# print('\n\n\n\n\n')
	# psi = psi
	# npsi = npsi
	#print(psi.shape, npsi.shape)
	pickle.dump((psi, npsi,ls,nls,ls_lin, nls_lin,psi_lin,npsi_lin), f)
