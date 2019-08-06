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
		ls = np.log(ls)
		nls = np.log(nls)
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
		low, not_low = _enlarge_feature(low, not_low, mode = 'log')
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

# may use tSNE or UMAP
def plot_hidden(file, algo, alpha, dimension):

	with open(file, 'rb') as f:
		(low, not_low) = pickle.load(f)

	low_num = low.shape[0]
	not_low_num = not_low.shape[0]

	total = np.concatenate((low, not_low), axis=0)
	print('total size', total.shape)

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
		title = 'Lambda_hidden_' + algo + '_2d.png'
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

'''
plot_tSNE(
	file = './lambda_vectors_minutes.pkl', 
	algo = 'tSNE', 
	alpha = 0.7,
	dimension = 2, 
	scale = 1, 
	enlarge = True)
'''

plot_hidden(
	file = './best_representation.pkl', 
	algo = 'tSNE', 
	alpha = 0.7,
	dimension = 2)

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



