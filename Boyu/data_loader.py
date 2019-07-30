import datetime
from datetime import date, datetime, time
import json
import argparse
import os
import numpy as np 
import pickle
from io import open

class data_loader(object):

	def __init__(self, file_path, lag, interval, category_one_hot_dict):
		print('data loader for {}'.format(file_path))
		self.file_path = file_path
		self.lag = lag

		# the interval that all searches within it will be sum up
		self.interval = interval

		# categories
		self.num_categories = len(category_one_hot_dict.keys())
		self.category_one_hot_dict = category_one_hot_dict
		assert lag < interval

	def parse(self):

		# tensors for each person
		tensors_all = []
		file_num = 0

		# per person
		for file in os.scandir(self.file_path):
			if file.name.endswith('.json'):
				file_num += 1
				print('file: {} {}'.format(self.file_path, file.name))

				whole_name = self.file_path + file.name
				# print(file.name)

				# list of all interval tensors for this person
				tensors_list = []

				# individual interval tensor (sum of all searches within it)
				tensor = np.zeros(self.num_categories)

				# previous timestamp, first place holder
				previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))

				# the current interval length
				current_length = 0

				# per search history
				with open(whole_name, 'r') as json_file:  
					search_history = [json.loads(line) for line in json_file]
					for instance in search_history:

						# print(instance['qtime'])
						date_time = self._trim_date(instance['qtime'])

						# get the time difference in seconds
						delta = (previous - date_time).total_seconds()

						# if the next search is out of interval
						if current_length + delta > self.interval:
							tensors_list.append(tensor)

							# start a new interval
							tensor = self._get_tensor(instance['category'])
							current_length = 0

						# continue within one interval
						elif delta > 0:

							# get the frequency tensor and add to the interval tensor
							# print('in')
							tensor += self._get_tensor(instance['category'])
							current_length += delta

						# the first dummy placeholder will be < 0
						# thus do not store it to the list
						else:
							tensor = self._get_tensor(instance['category'])

						# print(delta)
						previous = date_time

					# store the last tensor
					tensors_list.append(tensor)

				# reshape the matrix
				# np.stack(tensors_list): [self.num_category, num_interval]
				tensors_all.append(np.transpose(np.stack(tensors_list)))
				print('interval vectors for this person: {}\n'.format(tensors_all[-1].shape))

		assert len(tensors_all) == file_num
		# print(tensors_all[0].shape)

		if 'not' in self.file_path:
			p = self.file_path + 'not_low_freq_tensors.pkl'
		else:
			p = self.file_path + 'low_freq_tensors.pkl'

		with open(p, mode = 'wb') as f:
			pickle.dump(tensors_all, f)

	# parse the frequency vector by calendar month
	def parse_by_month(self):

		# tensors for each person
		tensors_all = []
		file_num = 0

		# per person
		for file in os.scandir(self.file_path):
			if file.name.endswith('.json'):
				file_num += 1
				print('file: {} {}'.format(self.file_path, file.name))

				whole_name = self.file_path + file.name

				# list of all interval tensors for this person
				tensors_list = []

				# individual interval tensor (sum of all searches within it)
				tensor = np.zeros(self.num_categories)

				# previous timestamp, first place holder
				previous = datetime.combine(date(2000, 1, 1), time(00, 00, 00))
				current_month = previous.month

				# per search history
				with open(whole_name, 'r') as json_file:  
					search_history = [json.loads(line) for line in json_file]
					for instance in search_history:

						# print(instance['qtime'])
						date_time = self._trim_date(instance['qtime'])

						# if enters a new month 
						# or remove the first dummy placeholder
						if current_month != date_time.month or previous == datetime.combine(date(2000, 1, 1), time(00, 00, 00)):

							# store the freq vector
							tensors_list.append(tensor)

							# start a new interval
							tensor = self._get_tensor(instance['category'])
							current_month = date_time.month

						# if within the same month
						else:

							# get the frequency tensor and add to the interval tensor
							# print('in')
							tensor += self._get_tensor(instance['category'])

						# print(delta)
						previous = date_time

					# store the last tensor
					tensors_list.append(tensor)

				# remove the first placeholder
				tensors_list = tensors_list[1:]

				# reshape the matrix
				# np.stack(tensors_list): [self.num_category, num_interval]
				tensors_all.append(np.transpose(np.stack(tensors_list)))
				print('interval vectors for the current person: {}\n'.format(tensors_all[-1].shape))

		assert len(tensors_all) == file_num
		# print(tensors_all[0].shape)

		if 'not' in self.file_path:
			p = self.file_path + 'not_low_freq_tensors_calendar.pkl'
		else:
			p = self.file_path + 'low_freq_tensors_calendar.pkl'

		with open(p, mode = 'wb') as f:
			pickle.dump(tensors_all, f)


	# helper function
	# generate the date time instance from the raw string
	def _trim_date(self, qtime):

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

	# helper method: get the frequency tensor for one search
	def _get_tensor(self, category_list):
		current_tensor = np.zeros(self.num_categories)
		for c in category_list:
			category = c[0].split('/')[1]

			# weighted one-hot vector
			current_tensor += self.category_one_hot_dict[category] * c[1]

		return current_tensor

# test run
def main():

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

	l2 = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]
	print(set(l).difference(set(l2)))

	# one-hot encoding for each category
	d = dict()
	for idx, category in enumerate(l2):
		arr = np.zeros(len(l2))
		arr[idx] = 1
		d.update({category: arr})
	# print(d)

	# one loader for each group
	nls_loader = data_loader(
		file_path = '/Users/mac/Downloads/Campus Study Data V1/original-data/low_self_esteem/', 
		lag = 1, 
		interval = 72000, 
		category_one_hot_dict = d)

	ls_loader = data_loader(
		file_path = '/Users/mac/Downloads/Campus Study Data V1/original-data/not_low_self_esteem/', 
		lag = 1, 
		interval = 72000, 
		category_one_hot_dict = d)

	# nls_loader.parse()
	nls_loader.parse_by_month()
	# ls_loader.parse()
	ls_loader.parse_by_month()

if __name__ == '__main__':
	main()