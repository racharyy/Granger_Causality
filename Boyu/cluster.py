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


# given all the user files in json
# convert the time stamps between adjacent search histories into time difference (delta)
def _get_delta_time(args):

	# the time difference list for all people
	total_delta_list = []

	# per person
	for file in os.scandir(args.data_path):
		if file.name.endswith('.json'):
			whole_name = args.data_path + file.name
			# print(file.name)

			# previous timestamp
			# first place holder
			previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))
		
			# the time difference list per person
			delta_list = []

			# per search history
			with open(whole_name, 'r') as json_file:  
				search_history = [json.loads(line) for line in json_file]

				for instance in search_history:

					# print(instance['qtime'])
					date_time = _trim_date(instance['qtime'])

					# get the time difference in minutes
					delta = (previous - date_time).total_seconds()

					if delta < args.threshold and delta > args.lower_bound:
						delta_list.append(delta)

					# print(delta)
					previous = date_time

			# remove the first placeholder
			total_delta_list.append(delta_list[1:])

	# return the flatten list
	with open('./total_list.pkl', mode = 'wb') as f:
		final_list = [val for sublist in total_delta_list for val in sublist]
		print(len(final_list))
		pickle.dump(final_list, f)

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

# plot the histogram for the distribution of delta time interval
def plot_hist(args, file_path):

	with open(file_path, mode = 'rb') as f:
		total_delta_list = pickle.load(f)

	# plot the raw version to get the log bins
	f = plt.figure(1)
	hist, bins, _ = plt.hist(total_delta_list, bins = 'auto')
	plt.xlabel('Seconds')
	plt.ylabel('Frequency')
	title = 'Idle Time Between Searches (upper: ' + str(args.threshold) + ', lower: ' + str(args.lower_bound) + ')'
	plt.title(title)
	name = '../../upper_' + str(args.threshold) + '_lower_' + str(args.lower_bound) + '_abs'
	plt.show()
	f.savefig(name)

	# get the log scale plot
	logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), args.bin_size)
	g = plt.figure(2)
	plt.hist(total_delta_list, bins = logbins, density = False)
	plt.xscale('log')

	plt.xlabel('Seconds (log 10 based)')
	plt.ylabel('Frequency')
	title = 'Idle Time Between Searches (upper: ' + str(args.threshold) + ', lower: ' + str(args.lower_bound) + ')'
	plt.title(title)
	plt.show()
	name = '../../upper_' + str(args.threshold) + '_lower_' + str(args.lower_bound)
	g.savefig(name)


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
						if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
							print(whole_name, idx, previous, date_time)

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
									if delta < 0 and previous != datetime.combine(date(2000, 1, 1), time(00, 00, 00)):
										print(whole_name, idx, previous, date_time)

									# print(delta)
									delta_list.append(delta)
									previous = date_time

					# fit the exponential distribution for each category
					# remove the first place holder
					if len(delta_list[1:]) > 1:
						param = _fit_exp_per_person(delta_list[1:])
						param_list.append(param)
					'''
					else:
						print(group, file.name, category, 'empty')
					'''
					
			if len(param_list) == 0:
				print(path, category, 'param list empty')
			else:

				# remove outlier
				'''
				x_100_median = np.median(np.asarray(param_list)) * 100
				param_list = [p for p in param_list if p < x_100_median]
				'''

				if group == 'low_self_esteem/':
					all_param_dict['low'] = param_list
				else:
					all_param_dict['not_low'] = param_list
			# print(path, category, 'done')

		for key, value in all_param_dict.items():
			print(category, key, np.mean(np.asarray(value)), np.median(np.asarray(value)))

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
		# print(category, 'all done')


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

	# _get_delta_time(args)
	# plot_hist(args, './total_list.pkl')
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

	fit_exp_category(args, normed = True, categories = l)

	'''
	plt.figure(figsize = (16, 9))
	all_param_dict = dict()
	low = [1.1142918896083136e-06, 7.91804202656767e-06, 5.891356094011244e-06, 4.6009262237834666e-07, 4.7536074071603134e-07, 5.486761233136746e-06, 1.5380115656370372e-06, 3.6700165266318807e-06, 9.271777885250748e-07, 3.135677199071384e-06, 9.62782213479099e-06, 2.6791399609088363e-07, 3.7161757109582004e-06, 1.1499090421947625e-06, 1.0486534000115845e-05, 4.414343730424059e-06, 4.971621515751635e-06, 9.258860388275671e-07, 2.0320733792996853e-05, 2.5147137492730746e-06, 8.428417483458449e-07, 0.25, 4.864367305132299e-06, 1.8561179154589332e-06, 3.69462884234368e-06, 5.271674657686323e-06, 1.481136922260619e-06, 1.6811950036419494e-06, 3.1288902742211822e-06, 3.250287323806146e-07, 1.3685540963357778e-05, 9.412296111645844e-07, 1.0425500142057518e-06, 7.49712223345575e-07, 1.2272364661744837e-06, 1.5280165042939138e-06, 6.48904926807168e-06, 1.4137008000914877e-06, 6.2427545030548915e-06, 3.3816315698147386e-06, 9.370623654630318e-07, 7.947311693217815e-08, 3.5906128933088304e-06, 3.5710890732698923e-06, 5.891356094011244e-06, 1.9206111070752732e-05, 5.81695794891454e-06, 9.016637215789035e-06, 4.204321622195455e-06, 2.27498088936173e-06, 1.6821148775194555e-06]
	not_low = [3.0289521465043043e-06, 2.247627091440751e-06, 1.3033548703866734e-06, 1.115020598060746e-06, 2.9188859433966395e-07, 5.9714404619548495e-06, 5.8976657611733174e-06, 3.2948103296062355e-07, 1.937231401830799e-06, 1.0508268364360798e-06, 2.6308381587289896e-06, 4.911735970949361e-06, 1.527314383377578e-06, 5.893869099131931e-06, 2.993900539082012e-06, 2.2095098553025618e-05, 1.1445818765739486e-06, 1.937231401830799e-06, 1.0440486743585387e-05, 5.126952165365398e-06, 2.415051645154916e-06, 4.010518227875447e-06, 6.461944537481227e-07, 1.4306776324252854e-06, 1.0640795063181848e-06, 6.039782157640453e-07, 1.1891568634061295e-06, 1.121912636662983e-05, 1.609536642567496e-06, 1.8124519269289515e-06, 6.48857203519104e-06, 2.7959828638724268e-06, 6.613784462804947e-07, 2.7629226933029145e-06, 9.915259606313298e-06, 1.3208526145894458e-07, 1.1832777702002867e-06, 2.7463074514147864e-06]
	all_param_dict['low'] = low
	all_param_dict['not_low'] = not_low
	for key, value in all_param_dict.items():
		plt.hist(value, bins = 'auto', alpha = 0.5, label = key)	
	plt.xlabel('Lambda')
	plt.ylabel('Frequency')
	plt.legend(loc = 'best')
	plt.title('title')
	plt.savefig('title', format='png', dpi=150, bbox_inches='tight')
	plt.close()
	'''

if __name__ == '__main__':
	main()
