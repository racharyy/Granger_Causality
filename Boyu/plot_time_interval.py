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

if __name__ == '__main__':
	main()
