import pickle
import glob
import json
import csv
# 216 categories
# [u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate', u'Business & Industrial', u'Home & Garden', u'Travel', u'Arts & Entertainment', u'Sports', u'Food & Drink', u'Pets & Animals', u'Health', u'Shopping', u'Finance', u'Adult', u'Beauty & Fitness', u'News', u'Books & Literature', u'Online Communities', u'Law & Government', u'Sensitive Subjects', u'Science', u'Hobbies & Leisure', u'Games', u'Jobs & Education', u'Autos & Vehicles', u'Computers & Electronics', u'People & Society', u'Reference', u'Internet & Telecom', u'Real Estate']

def createTrainingCatDist():
	all_cats = set(pickle.load(open("all_categories.pickle","rb")))
	train_lsFiles = glob.glob("train-low-self-esteem/*.json")

	ls_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_lsFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_lsFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1

			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
				

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		ls_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('ls_cat_search_percent_dist_by_users.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in ls_cat_search_percent_dist_by_users:
			writer.writerow(row)


	train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")

	nls_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_nlsFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_nlsFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1

			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
				

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		nls_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('nls_cat_search_percent_dist_by_users.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in nls_cat_search_percent_dist_by_users:
			writer.writerow(row)



	train_siFiles = glob.glob("train-suicide/*.json")

	si_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_siFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_siFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1
			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
			

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		si_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('si_cat_search_percent_dist_by_users.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in si_cat_search_percent_dist_by_users:
			writer.writerow(row)


	train_nsiFiles = glob.glob("train-non-suicide/*.json")
	nsi_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_nsiFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_nsiFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1
			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
			

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		nsi_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('nsi_cat_search_percent_dist_by_users.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in nsi_cat_search_percent_dist_by_users:
			writer.writerow(row)





def createTestCatDist():
	all_cats = set(pickle.load(open("all_categories.pickle","rb")))
	train_lsFiles = glob.glob("validation-low-self-esteem/*.json")

	ls_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_lsFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_lsFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1

			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
				

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		ls_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('ls_cat_search_percent_dist_by_users_validation.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in ls_cat_search_percent_dist_by_users:
			writer.writerow(row)

	
	train_nlsFiles = glob.glob("validation-not-low-self-esteem/*.json")

	nls_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_nlsFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_nlsFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1

			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
				

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		nls_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('nls_cat_search_percent_dist_by_users_validation.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in nls_cat_search_percent_dist_by_users:
			writer.writerow(row)



	train_siFiles = glob.glob("validation-suicide/*.json")

	si_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_siFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_siFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1
			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
			

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		si_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('si_cat_search_percent_dist_by_users_validation.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in si_cat_search_percent_dist_by_users:
			writer.writerow(row)


	train_nsiFiles = glob.glob("validation-non-suicide/*.json")
	nsi_cat_search_percent_dist_by_users = []
	header = ['']
	for file in train_nsiFiles:
		header.append(file.split("/")[-1].split(".")[0])

	for index,category in enumerate(all_cats):
		print index,'processing category', category
		percent_searches_for_each_cat_by_ls_users = [category]
		for file in train_nsiFiles:
			# per person levels
			total_searches = 0.0
			cat_counter_per_person = 0.0

			with open(file, 'r') as f:
				for line in f.readlines():
					data = json.loads(line)
					
					if data['category'] != []:
						top_cat = data['category'][0][0].split('/')[1]
						if top_cat == category:
							cat_counter_per_person+=1
						total_searches+=1
			if total_searches == 0.0:
				percent_searches_in_a_cat = 0.0
			else:
				percent_searches_in_a_cat = cat_counter_per_person/total_searches
			

			percent_searches_for_each_cat_by_ls_users.append(percent_searches_in_a_cat)

		# print percent_searches_for_each_cat_by_ls_users
		nsi_cat_search_percent_dist_by_users.append(percent_searches_for_each_cat_by_ls_users)


	with open('nsi_cat_search_percent_dist_by_users_validation.csv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter=',')
		writer.writerow(header)
		for row in nsi_cat_search_percent_dist_by_users:
			writer.writerow(row)


createTestCatDist()