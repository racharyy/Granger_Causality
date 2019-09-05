import datetime
from datetime import date, datetime, time
import json
import argparse
import os
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st

from io import open
import pickle
import sys
sys.path.insert(0, '../zipped_files_from_www')
from helper import *

# helper function
# generate the date time instance from the raw string
def _trim_date(qtime):

	date_list = qtime.replace(',', '').split(' ')
	date = datetime.strptime("{} {} {}".format(date_list[2], date_list[0], date_list[1]), "%Y %b %d")
	# print(date)
	time_raw = [int(x) for x in date_list[3].split(':')]

	# convert time format
	if 'PM' in date_list and time_raw[0] != 12:
		time_raw[0] += 12
	if 'AM' in date_list and time_raw[0] == 12:
		time_raw[0] = 00

	new_time = time(time_raw[0], time_raw[1], time_raw[2])
	# print(new_time)

	return datetime.combine(date, new_time)


# fit the data distribution to exponential distribution
def fit_exp(args, normed):

	# for both groups
	all_param_dict = {'low':[], 'not_low':[]}
	for group in ['low_self_esteem/', 'not_low_self_esteem/']:
		path = args.data_path + group
		param_list = []

		# per peron 
		for file in os.scandir(path):
			if file.name.endswith('.json'):

				whole_name = path + file.name
				# print(file.name)

				# previous timestamp, first place holder
				previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

				# the time difference list per person
				delta_list = []

				# per search history
				with open(whole_name, 'r') as json_file:  
					search_history = [json.loads(line) for line in json_file]

					for idx, instance in enumerate(search_history):

						# print(instance['qtime'])
						date_time = _trim_date(instance['qtime'])

						# get the time difference in minutes
						delta = (previous - date_time).total_seconds()

						# error report
						'''
						if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
							print(whole_name, idx, previous, date_time)
						'''

						delta_list.append(delta)
						previous = date_time

				# fit the exponential distribution
				# remove the first place holder
				param = _fit_exp_per_person(delta_list[1:])
				param_list.append(param)

		if group == 'low_self_esteem/':
			all_param_dict['low'] = param_list
		else:
			all_param_dict['not_low'] = param_list

	if normed:
		for key, value in all_param_dict.items():
			all_param_dict[key] = [val * 100000 for val in all_param_dict[key]]
			plt.hist(all_param_dict[key], bins = 'auto', alpha = 0.5, label = key, density = True)
			print('len for {}: {}'.format(key, len(value)))

			plt.xlabel('Lambda * 100000')
			title = 'Lambda Distribution (normed)'
	else:
		for key, value in all_param_dict.items():
			plt.hist(all_param_dict[key], bins = 'auto', alpha = 0.5, label = key)
			print('len for {}: {}'.format(key, len(value)))

			plt.xlabel('Lambda')
			title = 'Lambda Distribution'

	plt.ylabel('Frequency')
	plt.legend(loc = 'best')
	plt.title(title)
	plt.savefig(title)
	plt.show()

# helper method: get the lambda for each peron by fitting exponential distribution
def _fit_exp_per_person(delta_list):
	return 1 / np.mean(np.asarray(delta_list))

# get the mean and median lambda for exponential distribution
# based on each category
def mean_lambda_category(args, normed, categories):

	# for each categories
	for i, category in enumerate(categories):

		# for both groups
		all_param_dict = {'low':[], 'not_low':[]}
		for group in ['low_self_esteem/', 'not_low_self_esteem/']:
			path = args.data_path + group
			param_list = []

			# per peron 
			for file in os.scandir(path):
				if file.name.endswith('.json'):

					whole_name = path + file.name
					# print(file.name)

					# previous timestamp, first place holder
					previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

					# the time difference list per person
					delta_list = []

					# per search history
					with open(whole_name, 'r') as json_file:  
						search_history = [json.loads(line) for line in json_file]
						for idx, instance in enumerate(search_history):

							# only record the given category
							appeared_cat = set()
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category and current not in appeared_cat:

									# do not count duplicate
									appeared_cat.add(current)

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 0:
						param = _fit_exp_per_person(delta_list[1:])
						param_list.append(param)
					else:
						print(group, file.name, category, 'only searched once')
					
			if group == 'low_self_esteem/':
				all_param_dict['low'] = param_list
			else:
				all_param_dict['not_low'] = param_list
			# print(path, category, 'done')

		for key, value in all_param_dict.items():
			print(category, key, np.mean(np.asarray(value)), np.median(np.asarray(value)))


# fit the data distribution to exponential distribution
# based on each category
def fit_exp_category(args, normed, categories):

	# for each categories
	for i, category in enumerate(categories):

		# for both groups
		all_param_dict = {'low':[], 'not_low':[]}
		for group in ['low_self_esteem/', 'not_low_self_esteem/']:
			path = args.data_path + group
			param_list = []

			# per peron 
			for file in os.scandir(path):
				if file.name.endswith('.json'):

					whole_name = path + file.name
					# print(file.name)

					# previous timestamp, first place holder
					previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

					# the time difference list per person
					delta_list = []

					# per search history
					with open(whole_name, 'r') as json_file:  
						search_history = [json.loads(line) for line in json_file]
						for idx, instance in enumerate(search_history):

							# only record the given category
							appeared_cat = set()
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category and current not in appeared_cat:

									# do not count duplicate
									appeared_cat.add(current)

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 0:
						param = _fit_exp_per_person(delta_list[1:])
						param_list.append(param)
					'''
					else:
						print(group, file.name, category, 'only searched once')
					'''

			# remove outlier by 100 * median
			x_100_median = np.median(np.asarray(param_list)) * 100
			old_len = len(param_list)
			param_list = [p for p in param_list if p < x_100_median]
			if old_len != len(param_list):
				print('outlier:', old_len - len(param_list), path, category)

			if group == 'low_self_esteem/':
				all_param_dict['low'] = param_list
			else:
				all_param_dict['not_low'] = param_list
			# print(path, category, 'done')

		# plot for each category
		'''
		plt.figure(i, figsize = (16, 9))
		if normed:
			for key, value in all_param_dict.items():
				weights = np.ones_like(value) / float(len(value))
				plt.hist(value, bins = 30, alpha = 0.5, label = key, weights = weights)
				title = 'Lambda Distribution (normed) ' + category
		else:
			for key, value in all_param_dict.items():
				plt.hist(value, bins = 'auto', alpha = 0.5, label = key)
				title = 'Lambda Distribution ' + category
		plt.xlabel('Lambda')
		plt.ylabel('Frequency')
		plt.legend(loc = 'best')
		plt.title(title)
		plt.savefig(title, format='png', dpi=150, bbox_inches='tight')
		plt.close()
		'''
		print(category, 'all done')


# find CV
def find_CV(args, normed, categories):

	# for each categories
	ls_cv, nls_cv = [], []
	for i, category in enumerate(categories):

		# for both groups
		# all_CV_dict = {'low':[], 'not_low':[]}

		for group in ['low_self_esteem/', 'not_low_self_esteem/']:
			path = args.data_path + group

			mean_list = []
			std_list = []

			# per peron 
			for file in os.scandir(path):
				if file.name.endswith('.json'):

					whole_name = path + file.name
					# print(file.name)

					# previous timestamp, first place holder
					previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

					# the time difference list per person
					delta_list = []

					# per search history
					with open(whole_name, 'r') as json_file:  
						search_history = [json.loads(line) for line in json_file]
						for idx, instance in enumerate(search_history):

							# only record the given category
							appeared_cat = set()
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category and current not in appeared_cat:

									# do not count duplicate
									appeared_cat.add(current)

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 0:

						mean = np.mean(delta_list[1:])
						std = np.std(delta_list[1:])
						# param = _fit_exp_per_person(delta_list[1:])
						mean_list.append(mean)
						std_list.append(std)
					'''
					else:
						print(group, file.name, category, 'only searched once')
					'''
					
			# remove outlier by 100 * median
			'''
			x_100_median = np.median(np.asarray(param_list)) * 100
			old_len = len(param_list)
			param_list = [p for p in param_list if p < x_100_median]
			if old_len != len(param_list):
				print('outlier:', old_len - len(param_list), path, category)
			'''
			assert len(std_list) == len(mean_list)
			CV = np.mean(np.asarray(std_list) / np.asarray(mean_list))
			if group == 'low_self_esteem/':
				ls_cv.append(CV)
			else:
				nls_cv.append(CV)
			print(group, category, CV)
	return ls_cv,nls_cv


# extract the lambda feature vector for each person
# 27-d category vector for each person
# with user id reference
def extract_lambda_feature_with_ID(args, categories, outlier, outlier_scale, scale):

	# for each categories; size of [27, number of people]
	low_matrix = []
	not_low_matrix = []

	# keep track of user id
	low_user_id = []
	not_low_user_id = []

	# for both groups
	for group in ['low_self_esteem/', 'not_low_self_esteem/']:
		path = args.data_path + group

		# per peron 
		for file in os.scandir(path):
			if file.name.endswith('.json'):

				user_id = file.name.split('.')[0]

				# 27-d vector for the current user
				category_vector = [] 

				# for each categories
				for i, category in enumerate(categories):

					whole_name = path + file.name

					# previous timestamp, first place holder
					previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

					# the time difference list per person
					delta_list = []

					# per search history
					with open(whole_name, 'r') as json_file:  
						search_history = [json.loads(line) for line in json_file]
						for idx, instance in enumerate(search_history):

							# only record the given category
							appeared_cat = set()
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category and current not in appeared_cat:

									# do not repeat category since google gives more details
									appeared_cat.add(current)

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds() / scale

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 0:
						param = _fit_exp_per_person(delta_list[1:])
						category_vector.append(param)
					else:
						# print(group, file.name, category, 'only searched once')
						category_vector.append(0)

				# verify output and store with user id
				assert len(category_vector) == 27
				if group == 'low_self_esteem/':
					low_matrix.append(category_vector)
					low_user_id.append(user_id)
				else:
					not_low_matrix.append(category_vector)
					not_low_user_id.append(user_id)
		print(group, 'done')

	# [27, number of people] after transpose
	low_matrix = np.stack(low_matrix).T
	not_low_matrix = np.stack(not_low_matrix).T

	# eliminate outlier
	if outlier:

		# for each group [27, number of people]
		for matrix in [low_matrix, not_low_matrix]:

			# for each category; each row is [number of people]
			for idx, row in enumerate(matrix):

				# hard threshold based on scle of median
				threshold = np.median(row) * outlier_scale
				row[row >= threshold] = threshold
				matrix[idx] = row

	low_list = [(user_id, low_matrix[:, idx]) for idx, user_id in enumerate(low_user_id)]
	not_low_list = [(user_id, not_low_matrix[:, idx]) for idx, user_id in enumerate(not_low_user_id)]
	print('low shape: [{} {}], not low shape: [{} {}]'.format(len(low_list), low_list[0][1].shape, len(not_low_list), not_low_list[0][1].shape))
	print('examples {} {}'.format(low_list[0][1], not_low_list[0][1]))

	# return list[tuple(user_id, lambda numpy vector)]
	if outlier:
		file_name = './lambda_vectors_cleaned_' + str(scale) + '.pkl'
	else:
		file_name = './lambda_vectors_' + str(scale) + '.pkl'

	with open(file_name, mode = 'wb') as f:
		pickle.dump((low_list, not_low_list), f)		


def _verify(path):

	with open('./lambda_vectors_minutes.pkl', mode = 'rb') as f:
		(low_list, not_low_list) = pickle.load(f)
	print('low shape: [{} {}], not low shape: [{} {}]'.format(len(low_list), low_list[0][1].shape, len(not_low_list), not_low_list[0][1].shape))

	print('first sample: {} {}'.format(low_list[0][1], not_low_list[0][1]))

	low_list = np.stack([user[1] for user in low_list])
	not_low_list = np.stack([user[1] for user in not_low_list])

	plot_lambda(low_list, not_low_list)


# generate compound feature vector with the WWW 2019 paper
# for both the NLS/LS and PSI groups
def generate_compound_features_with_ID(args, scaled, cat_ls_path, cat_nls_path, lambda_path):

	# load categorial vectors
	cat_ls_list = load_pickle(cat_ls_path) # 51
	cat_nls_list = load_pickle(cat_nls_path) # 45

	# remove invalid data
	spanish = ['360472c2621eaa', '3651870e0399e4', '3656c6f2301a24']
	cat_ls_list = [t for t in cat_ls_list if t[0] not in spanish]
	cat_nls_list = [t for t in cat_nls_list if t[0] not in spanish]

	# load lambda features
	with open(lambda_path, 'rb') as f3:
		(low_list, not_low_list) = pickle.load(f3)

	# check duplicate '35ffb55217e592'
	import collections
	a = [t[0] for t in cat_ls_list]
	print([item for item, count in collections.Counter(a).items() if count > 1])
	print(len(low_list), len(not_low_list)) # 54 38
	print(len(cat_ls_list), len(cat_nls_list))

	# convert to dict
	low_list = [(e[0].split('_')[1], e[1]) for e in low_list]
	not_low_list = [(e[0].split('_')[1], e[1]) for e in not_low_list]
	cat_low_dict = dict()
	cat_not_low_dict = dict()
	cat_low_dict.update(cat_ls_list)
	cat_not_low_dict.update(cat_nls_list)

	# scale up first
	low = np.stack([user[1] * scaled for user in low_list])
	not_low = np.stack([user[1] * scaled for user in not_low_list])

	# generate the compound data
	compound_ls_list = []
	for idx, user_data in enumerate(low_list):
		vector = np.concatenate((cat_low_dict.get(user_data[0]), low[idx]))
		compound_ls_list.append((user_data[0], vector))

	compound_nls_list = []
	for idx, user_data in enumerate(not_low_list):
		vector = np.concatenate((cat_not_low_dict.get(user_data[0]), not_low[idx]))
		compound_nls_list.append((user_data[0], vector))

	print('low shape: [{} {}], not low shape: [{} {}]'.format(len(compound_ls_list), compound_ls_list[0][1].shape, len(compound_nls_list), compound_nls_list[0][1].shape))
	with open('./compound_vectors_self_esteem.pkl', mode = 'wb') as f:
		pickle.dump((compound_ls_list, compound_nls_list), f)	

	# for PSI data mapping
	all_psi_users = ['35808080808080', '3603a2c8d630e8', '360f70686ed3c2', '35f88eac597396', '36047578a646d8', '36103c719a02b8', '35f96084a59be8', '3605480ff725aa', '3610e92d3ff426', '35f98283c97fa4', '3606012f7da94c', '35fd53a205f032', '360ea61b204b08', '35818181818181', '35fa42ba5190a4', '360ea7d4ca7b3e', '35f955ff00cba2', '360e98e333ab46', '364d68a4996976', '36517e923dda2c', '3651870e0399e4', '3652ed3ac0fb96', '3656c6f2301a24', '365de175913efe', '3661cbdf582378']
	all_npsi_users = ['35777777777777', '35fed6a9a0bf1c', '360ddfbd854066', '35787878787878', '35fed7c5974de0', '360e93d13eabc4', '35797979797979', '35fed90a9f8324', '360e9521584ce6', '35fa3211bd0a02', '35fee27f6a26ca', '360eacde42f07c', '35fa42ba5190a4', '35fef442aba2f8', '360f61a2ebff78', '35faf634ba1e9e', '35ffa65213cd1a', '36102f386840b2', '35fafa696f1180', '35ffb55217e592', '36103b2f43f810', '35fb0466ef53b6', '3600808bcb8f78', '361044da6b5d9a', '35fb0a1ac2e81c', '360088be33e584', '36104a968d9ee2', '35fd4f85e6ad56', '360472c2621eaa', '3611064f3a7244', '35fd5b83b88b12', '360518c09b9ff4', '36110ae10c3d3e', '35fd5d66c7e2f0', '36051ecd8c4ca2', '3626f6903354b2', '35fe072bb73ce2', '3605f60c751daa', '3626f804162108', '35fe13b4df2dc8', '3605fc6be3d63c', '35f9679e778d20', '35ff04767aa900', '36103eac87dde6', '35f9891dbcb4bc', '3602d0b7f8f7de', '361042a326fdc0', '35fa37921286fe', '3603b99b42d928', '3610ffb257067e', '35fafcced91e3a', '360521efaa3e7e', '35fd568f77d0de', '360e9006cc0942', '365c81adeeeb2e', '365c866bbabc36', '3665f2fe0de374', '3665f8af1aef52', '36674fded9458e', '3667533c957a12', '36675399d93e54', '366e83536e78e2', '367d82e66579f4', '368e04e9e9a792']
	print(len(all_psi_users), len(all_npsi_users))

	# clean up invalid and duplicate data
	all_psi_users = set([u for u in all_psi_users if u not in spanish])
	all_npsi_users = set([u for u in all_npsi_users if u not in spanish])
	all_coumpound_dict = dict()
	all_coumpound_dict.update(compound_nls_list + compound_ls_list)

	psi = [(user, all_coumpound_dict.get(user)) for user in all_psi_users]
	npsi = [(user, all_coumpound_dict.get(user)) for user in all_npsi_users]

	print('psi shape: [{} {}], npsi shape: [{} {}]'.format(len(psi), psi[0][1].shape, len(npsi), npsi[0][1].shape))
	with open('./compound_vectors_psi.pkl', mode = 'wb') as f:
		pickle.dump((psi, npsi), f)

def _verify_cat(cat_ls_path, cat_nls_path, my_path):

	with open('./compound_vectors_psi.pkl', mode = 'rb') as f:
		psi, npsi = pickle.load(f)

	for user in psi:
		if user[0] == '365705c34176a4':
			print(user)

	for user in npsi:
		if user[0] == '365705c34176a4':
			print(user)
	'''
	# load categorial vectors
	cat_ls_list = load_pickle(cat_ls_path) # 51
	cat_nls_list = load_pickle(cat_nls_path) # 45

	# remove invalid data
	spanish = ['360472c2621eaa', '3651870e0399e4', '3656c6f2301a24']
	cat_ls_list = [t for t in cat_ls_list if t[0] not in spanish]
	cat_nls_list = [t for t in cat_nls_list if t[0] not in spanish]

	with open(my_path, 'rb') as f:
		compound_ls_list, compound_nls_list = pickle.load(f)

	for user in compound_ls_list:
		for u in cat_ls_list:
			if user[0] == u[0]:
				# print('ls', user[0])
				# print()
				assert np.array_equal(user[1][:27], u[1])

	for user in compound_nls_list:
		for u in cat_nls_list:
			if user[0] == u[0]:
				# print('nls', user[0])
				assert np.array_equal(user[1][:27], u[1])

	plot_lambda(compound_ls_list, compound_nls_list)
	'''

def plot_lambda(ls_list, nls_list):
	
	low_mean = np.log(np.mean(np.array(ls_list),axis=0))
	notlow_mean = np.log(np.mean(np.array(nls_list),axis=0))
	# print(low_mean)
	# print(notlow_mean)
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

def main():
	parser = argparse.ArgumentParser(description = 'parser for data files')
	parser.add_argument('--data_path', metavar = 'D', type = str, nargs = 1, 
						default = '/Users/mac/Downloads/Campus Study Data V1/original-data/',
						help = 'data file path')
	parser.add_argument('--cluster_num', default = 2,
						help = 'number of clusters')
	parser.add_argument('--bin_size', default = 0,
						help = 'bin size for the inter-time histogram')
	parser.add_argument('--threshold', default = 100000000,
						help = 'the max delta time between two searches in minutes')
	parser.add_argument('--lower_bound', default = 0,
						help = 'the min delta time between two searches in minutes')

	args = parser.parse_args()

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

	# fit_exp_category(args, normed = True, categories = l)
	# extract_lambda_feature(args, categories = l, outlier = True, outlier_scale = 100, normed = True)

	# for PSI data mapping
	'''
	all_psi_users = ['35808080808080', '3603a2c8d630e8', '360f70686ed3c2', '35f88eac597396', '36047578a646d8', '36103c719a02b8', '35f96084a59be8', '3605480ff725aa', '3610e92d3ff426', '35f98283c97fa4', '3606012f7da94c', '35fd53a205f032', '360ea61b204b08', '35818181818181', '35fa42ba5190a4', '360ea7d4ca7b3e', '35f955ff00cba2', '360e98e333ab46', '364d68a4996976', '36517e923dda2c', '3651870e0399e4', '3652ed3ac0fb96', '3656c6f2301a24', '365de175913efe', '3661cbdf582378']
	all_npsi_users = ['35777777777777', '35fed6a9a0bf1c', '360ddfbd854066', '35787878787878', '35fed7c5974de0', '360e93d13eabc4', '35797979797979', '35fed90a9f8324', '360e9521584ce6', '35fa3211bd0a02', '35fee27f6a26ca', '360eacde42f07c', '35fa42ba5190a4', '35fef442aba2f8', '360f61a2ebff78', '35faf634ba1e9e', '35ffa65213cd1a', '36102f386840b2', '35fafa696f1180', '35ffb55217e592', '36103b2f43f810', '35fb0466ef53b6', '3600808bcb8f78', '361044da6b5d9a', '35fb0a1ac2e81c', '360088be33e584', '36104a968d9ee2', '35fd4f85e6ad56', '360472c2621eaa', '3611064f3a7244', '35fd5b83b88b12', '360518c09b9ff4', '36110ae10c3d3e', '35fd5d66c7e2f0', '36051ecd8c4ca2', '3626f6903354b2', '35fe072bb73ce2', '3605f60c751daa', '3626f804162108', '35fe13b4df2dc8', '3605fc6be3d63c', '35f9679e778d20', '35ff04767aa900', '36103eac87dde6', '35f9891dbcb4bc', '3602d0b7f8f7de', '361042a326fdc0', '35fa37921286fe', '3603b99b42d928', '3610ffb257067e', '35fafcced91e3a', '360521efaa3e7e', '35fd568f77d0de', '360e9006cc0942', '365c81adeeeb2e', '365c866bbabc36', '3665f2fe0de374', '3665f8af1aef52', '36674fded9458e', '3667533c957a12', '36675399d93e54', '366e83536e78e2', '367d82e66579f4', '368e04e9e9a792']
	print(len(all_psi_users), len(all_npsi_users))
	'''
	
	'''
	ls_cv,nls_cv=find_CV(args, normed = True, categories = l)
	pickle.dump((ls_cv,nls_cv),open('cv_list.pkl','wb'))
	'''

	'''
	extract_lambda_feature_with_ID(
		args = args, 
		categories = l, 
		outlier = False, 
		outlier_scale = 100, 
		scale = 60)
	'''
	
	# _verify('./lambda_vectors_minutes.pkl')
		
	'''
	generate_compound_features_with_ID(
		args = args,
		scaled = 10**5, 
		cat_ls_path = '../searchCatDistData/ls_category_vectors_with_user_ID.pkl', 
		cat_nls_path = '../searchCatDistData/nls_category_vectors_with_user_ID.pkl', 
		lambda_path = './lambda_vectors_with_user_ID.pkl')
	'''

	'''
	_verify_cat(
		cat_ls_path = '../searchCatDistData/ls_category_vectors_with_user_ID.pkl', 
		cat_nls_path = '../searchCatDistData/nls_category_vectors_with_user_ID.pkl', 
		my_path = './compound_vectors_self_esteem.pkl')
	'''

if __name__ == '__main__':
	main()
