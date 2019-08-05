import pickle
import numpy as np
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import math
import itertools
import umap

# enlarge the features by exp
# (49, 27) (43, 27)
def _enlarge_feature(ls, nls, mode, multiplier = 1):

	ls_mean, ls_var = np.mean(ls, axis = 0), np.var(ls, axis = 0)
	nls_mean, nls_var = np.mean(nls, axis = 0), np.var(nls, axis = 0)

	if mode == 'delta':
		assert multiplier > 0

		raw_delta = ls_mean - nls_mean
		delta_new = np.absolute(raw_delta) * multiplier
		for idx, d in enumerate(raw_delta):
			if d > 0:
				ls[:, idx] += delta_new[idx]
				nls[:, idx] -= delta_new[idx]
			elif d < 0:
				ls[:, idx] -= delta_new[idx]
				nls[:, idx] += delta_new[idx]

		ls_mean, ls_var = np.mean(ls, axis = 0), np.var(ls, axis = 0)
		nls_mean, nls_var = np.mean(nls, axis = 0), np.var(nls, axis = 0)

	else:

		ls = np.exp(ls)
		nls = np.exp(nls)
		
		ls = np.exp(ls)
		nls = np.exp(nls)
		for v in ls:
			print(v[3], v[7])
		print('mean: ', np.mean(ls[:, 3]), np.mean(ls[:, 7]))

	return ls, nls

# may use tSNE or UMAP
def plot_tSNE(file, algo, alpha, dimension, scale, enlarge):

	with open(file, 'rb') as f:
		(low_list, not_low_list) = pickle.load(f)

	# scale up first
	low = np.stack([user[1] * scale for user in low_list])
	not_low = np.stack([user[1] * scale for user in not_low_list])
	# plot_lambda(low, not_low, multiplier = 1)

	if enlarge:
		low, not_low = _enlarge_feature(low, not_low, mode = 'exp')
		plot_lambda(low, not_low)
	print('shapes', low.shape, not_low.shape)

	low_num = low.shape[0]
	not_low_num = not_low.shape[0]

	total = np.concatenate((low, not_low), axis=0)
	print('total size', total.shape)

	low_list = [(low_list[idx][0], vec) for idx, vec in enumerate(low)]
	not_low_list = [(not_low_list[idx][0], vec) for idx, vec in enumerate(not_low)]
	with open('lambda_enlarged', 'wb') as f:
		pickle.dump((low_list, not_low_list), f)

	# fit tSNE
	if dimension == 3: 

		if algo == 'umap':
			fit = umap.UMAP(
				n_neighbors = 25,
				min_dist = 0.1,
				n_components = 3,
				metric = 'euclidean')
			embeddings = fit.fit_transform(total)
		else:
			embeddings = TSNE(n_components = 3).fit_transform(total)
		print(embeddings.shape)

		fig = plt.figure(figsize=(16, 9))
		ax = fig.add_subplot(111, projection='3d')

		low_matrix = embeddings[:low_num, :]
		print(low_matrix.shape)
		x = low_matrix[:, 0]
		y = low_matrix[:, 1]
		z = low_matrix[:, 2]
		ax.scatter(x, y, z, alpha = alpha, label = 'low')

		not_low_matrix = embeddings[low_num:, :]
		print(not_low_matrix.shape)
		x = not_low_matrix[:, 0]
		y = not_low_matrix[:, 1]
		z = not_low_matrix[:, 2]
		ax.scatter(x, y, z, alpha = alpha, label = 'not_low')

		plt.legend(loc = 'best')
		title = 'Lambda Features_' + algo + '_3d'
		plt.title(title)
		plt.grid(True)
		plt.show()
		plt.savefig(title, format='png', dpi=150, bbox_inches='tight')

	else:

		if algo == 'umap':
			fit = umap.UMAP(
				n_neighbors = 50,
				min_dist = 0.1,
				n_components = 2,
				metric = 'euclidean')
			embeddings = fit.fit_transform(total)
		else:
			embeddings = TSNE(n_components = 2).fit_transform(total)
		print(embeddings.shape)

		fig = plt.figure(figsize=(16, 9))

		low_matrix = embeddings[:low_num, :]
		print(low_matrix.shape)
		x = low_matrix[:, 0]
		y = low_matrix[:, 1]
		plt.scatter(x, y, alpha = alpha, label = 'low')

		not_low_matrix = embeddings[low_num:, :]
		print(not_low_matrix.shape)
		x = not_low_matrix[:, 0]
		y = not_low_matrix[:, 1]
		plt.scatter(x, y, alpha = alpha, label = 'not_low')

		plt.legend(loc = 'best')
		title = 'Lambda Features_' + algo + '_2d'
		plt.title(title)
		plt.grid(True)
		# plt.show()
		plt.savefig(title, format='png', dpi=150, bbox_inches='tight')

def plot_lambda(ls_list, nls_list):
	
	low_mean = np.mean(np.array(ls_list), axis=0)
	notlow_mean = np.mean(np.array(nls_list), axis=0)

	print('low mean', low_mean)
	print('not low mean', notlow_mean)
	labels = ["c"+str(i+1) for i in range(27)]
	xaxis = np.arange(27)
	width = 0.3 

	fig, ax = plt.subplots()
	rects1 = ax.bar(xaxis - width/2, low_mean, width, label='Low')
	rects2 = ax.bar(xaxis + width/2, notlow_mean, width, label='Not Low')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('ISI params')
	ax.set_title('ISI for two groups')
	ax.set_xticks(xaxis)
	ax.set_xticklabels(labels)
	ax.legend()
	
	fig.tight_layout()
	plt.show()
	plt.close()

# (49, 27) (43, 27)
def lambda_hist(lambda_path, cats, multiplier = 10**5):

	with open(lambda_path, 'rb') as f:
		(low_list, not_low_list) = pickle.load(f)

	# scale up first
	low = np.stack([user[1] * multiplier for user in low_list])
	not_low = np.stack([user[1] * multiplier for user in not_low_list])
	plot_lambda(low, not_low)

	for idx, c in enumerate(cats):
		plt.figure(idx)

		ls = low[:, idx]
		nls = not_low[:, idx]

		plt.hist(ls, bins = 'auto', alpha = 0.5, label = 'low')
		plt.hist(nls, bins = 'auto', alpha = 0.5, label = 'not_low')
		plt.legend(loc='best')
		plt.savefig('../Plots/' + c + '_hist')
		plt.close()

plot_tSNE(
	file = './lambda_vectors_with_user_ID.pkl', 
	algo = 'tSNE', 
	alpha = 0.7,
	dimension = 2, 
	scale = 10**4, 
	enlarge = True)

l = [
"Business & Industrial",
"Home & Garden",
"Travel",
"Arts & Entertainment",
"Sports",
"Food & Drink",
"Pets & Animals",
"Health",
"Shopping",
"Finance",
"Adult",
"Beauty & Fitness",
"News",
"Books & Literature",
"Online Communities",
"Law & Government",
"Sensitive Subjects",
"Science",
"Hobbies & Leisure",
"Games",
"Jobs & Education",
"Autos & Vehicles",
"Computers & Electronics",
"People & Society",
"Reference",
"Internet & Telecom",
"Real Estate"]

'''
lambda_hist(
	lambda_path = './lambda_vectors_with_user_ID.pkl', 
	cats = l, 
	multiplier = 10**5)

'''

'''
b1 = [0.41323691, 0.10256799, 0.33442297, 1.87672492, 0.59377013, 0.32891397, 0.07773937, 0.83892633, 0.38159667, 0.19727683, 0.09810761, 0.17052776, 0.27473965, 0.2998026,  0.1646275,  0.18843744, 0.09216614, 0.51360599, 0.29093614, 0.30276502, 0.65227786, 0.18002815, 0.58324862, 0.40844325, 0.733122, 0.57482068, 0.06468569]
b2 = [0.30347907, 0.08550423, 0.20939409, 1.24217429, 0.1897198,  0.2654285, 0.17053899, 0.30057931, 0.31235085, 0.56038426, 0.07994414, 0.16532287, 0.22400172, 0.16320343, 0.14177989, 0.1575167,  0.09351142, 0.4587919, 0.21412693, 0.18938543, 0.6570595,  0.08136606, 0.46751621, 0.34257651, 0.72109435, 0.23860313, 0.02443967]
b1 = np.asarray(b1)
b2 = np.asarray(b2)

d1 = []
d2 = []

for i in range(100):
	d1.append(np.random.rand(27)  + b1)
	d2.append(np.random.rand(27)  + b2)

d1 = np.stack(d1)
d2 = np.stack(d2)	
d = np.concatenate((d1, d2), axis = 0)
print(d.shape)

d1_l = [v for v in d1]
d2_l = [v for v in d2]
plot_lambda(d1_l, d2_l, multiplier = 1)

embeddings = TSNE(n_components = 2).fit_transform(d)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
ax.scatter(embeddings[:100,0], embeddings[:100,1])
ax.scatter(embeddings[100:,0], embeddings[100:,1])
plt.show()
'''


