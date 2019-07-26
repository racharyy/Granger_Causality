import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

train_lsFiles = glob.glob("train-low-self-esteem/*.json")
train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")


def normalixe_dic(d):
	total = 0.0

	for k,v in d.iteritems():
		total += v

	for k,v in d.iteritems():
		d[k] = (v/total)*100.0


	return d

ltracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
mtracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
ntracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
etracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}



cat_code = {"Business & Industrial" : 0,
"Home & Garden" : 1,
"Travel" : 2,
"Arts & Entertainment" : 3,
"Sports" : 4,
"Food & Drink" : 5,
"Pets & Animals" : 6,
"Health" : 7,
"Shopping" : 8,
"Finance" : 9,
"Adult" : 10,
"Beauty & Fitness" : 11,
"News" : 12,
"Books & Literature" : 13,
"Online Communities" : 14,
"Law & Government" : 15,
"Sensitive Subjects" : 16,
"Science" : 17,
"Hobbies & Leisure" : 18,
"Games" : 19,
"Jobs & Education" : 20,
"Autos & Vehicles" : 21,
"Computers & Electronics" : 22,
"People & Society" : 23,
"Reference" : 24,
"Internet & Telecom" : 25,
"Real Estate" : 26
}

all_cat_sequences = {}

for file in train_nlsFiles:
	with open(file, 'r') as f:
		category_sequence = []
		lines = f.readlines()
		for index,line in enumerate(lines):

			next_index = index + 1

			if next_index < len(lines):
				data = json.loads(line)
				next_data = json.loads(lines[next_index])

				if data['category'] != [] and next_data['category'] != []:
					top_cat = data['category'][0][0].split('/')[1]
					next_top_cat = next_data['category'][0][0].split('/')[1]

					if top_cat == next_top_cat:
						key = top_cat+'->'+next_top_cat
						if key not in all_cat_sequences:
							all_cat_sequences[key] = 1
						else:
							all_cat_sequences[key] += 1

					else:
						

					print 
					# time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p') 
					# category_sequence.append(cat_code[top_cat])


		# print category_sequence
		# all_cat_sequences.append(category_sequence)

# all_cat_sequences = np.array(all_cat_sequences)
# print all_cat_sequences.shape

# np.savetxt('ls-cat-sequence.csv',all_cat_sequences)

