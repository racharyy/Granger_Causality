import datetime
from datetime import date, datetime, time
import json
import argparse
import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import pickle
from io import open
import scipy.stats as st


# helper function
# generate the date time instance from the raw string
def _trim_date(qtime):

	date_list = qtime.replace(',', '').split(' ')
	date = datetime.strptime("{} {} {}".format(date_list[2], date_list[0], date_list[1]), "%Y %b %d")
	# print(date)


	time_raw = [int(x) for x in date_list[3].split(':')]

	'''
	if date_list[-1] == 'PM' and time_raw[0] == 00:
		print(qtime)
	if date_list[-1] == 'AM' and time_raw[0] == 00:
		print(qtime)
	'''

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
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category:

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# error report
									'''
									if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
										print(whole_name, idx, previous, date_time)
									'''

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 1:
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
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category:

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# error report
									'''
									if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
										print(whole_name, idx, previous, date_time)
									'''

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 1:
						param = _fit_exp_per_person(delta_list[1:])
						param_list.append(param)
					else:
						print(group, file.name, category, 'only searched once')

			# remove outlier by 100 * median
			x_100_median = np.median(np.asarray(param_list)) * 100
			param_list = [p for p in param_list if p < x_100_median]

			if group == 'low_self_esteem/':
				all_param_dict['low'] = param_list
			else:
				all_param_dict['not_low'] = param_list
			print(path, category, 'done')

		# plot for each category
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
		print(category, 'all done')


# extract the lambda feature vector for each person
# 27-d category vector for each person
def extract_lambda_feature(args, categories):

	# for each categories
	# each is 95 * 27
	low_matrix = []
	not_low_matrix = []

	# for both groups
	for group in ['low_self_esteem/', 'not_low_self_esteem/']:
		path = args.data_path + group

		# per peron 
		for file in os.scandir(path):
			if file.name.endswith('.json'):

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
							for c in instance['category']:
								current = c[0].split('/')[1]

								if current == category:

									# print(instance['qtime'])
									date_time = _trim_date(instance['qtime'])

									# get the time difference in minutes
									delta = (previous - date_time).total_seconds()

									# error report
									'''
									if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
										print(whole_name, idx, previous, date_time)
									'''

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 1:
						param = _fit_exp_per_person(delta_list[1:])
						category_vector.append(param)
					else:
						print(group, file.name, category, 'only searched once')
						category_vector.append(0)

				# verify output and store away
				assert len(category_vector) == 27
				if group == 'low_self_esteem/':
					low_matrix.append(category_vector)
				else:
					not_low_matrix.append(category_vector)

		print(group, 'done')

	low_matrix = np.stack(low_matrix)
	not_low_matrix = np.stack(not_low_matrix)
	print('low shape: {}, not low shape: {}'.format(low_matrix.shape, not_low_matrix.shape))

	# return the 95 * 27 matrix
	with open('./lambda_data_per_user_matrix.pkl', mode = 'wb') as f:
		pickle.dump((low_matrix, not_low_matrix), f)

def main():
	parser = argparse.ArgumentParser(description = 'parser for data files')
	parser.add_argument('--data_path', metavar = 'D', type = str, nargs = 1, 
						default = '../original-data/',
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

	# fit_exp(args, normed = True)

	l = [
	'Health', 
	'Online Communities', 
	'Books & Literature', 
	'Food & Drink', 
	'Autos & Vehicles', 
	'Law & Government', 
	'Beauty & Fitness', 
	'Sports', 
	'Business & Industrial', 
	'Pets & Animals', 
	'Science', 
	'Real Estate', 
	'Jobs & Education', 
	'Games', 
	'Internet & Telecom', 
	'Home & Garden', 
	'Finance', 
	'Sensitive Subjects', 
	'Shopping', 
	'Arts & Entertainment', 
	'News', 
	'Reference', 
	'Computers & Electronics', 
	'Adult', 
	'Hobbies & Leisure', 
	'People & Society', 
	'Travel']

	# fit_exp_category(args, normed = True, categories = l)
	extract_lambda_feature(args, categories = l)

if __name__ == '__main__':
	main()
