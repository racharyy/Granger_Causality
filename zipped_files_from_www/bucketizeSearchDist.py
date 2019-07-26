import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

hr_index_bucket = {
	0 : 0,
	1 : 0,
	2 : 0,
	3 : 0,
	4 : 1,
	5 : 1,
	6 : 1,
	7 : 1,
	8 : 2,
	9 : 2,
	10 : 2,
	11 : 2,
	12 : 3,
	13 : 3,
	14 : 3,
	15 : 3,
	16 : 4,
	17 : 4,
	18 : 4,
	19 : 4,
	20 : 5,
	21 : 5,
	22 : 5,
	23 : 5,
}

# def bucketizeSearchDist(Files, target):

# 	for file in Files:
# 		with open(file, 'r') as f:
# 			header = ['','0-3', '4-7','8-11', '12-15','16-19', '19-23']
# 			print file
			
# 			all_cat_vectors = [header]

# 			for cat in ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]:
# 				row = [0.0]*len(header)
# 				row[0] = cat
# 				for line in f.readlines():
# 					data = json.loads(line)
# 					if data['category'] != []:
# 						top_cat = data['category'][0][0].split('/')[1]
# 						if cat == top_cat:
# 							time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p')
							
# 							# print top_cat, hr_index_bucket[int(time.hour)], time
# 							row[hr_index_bucket[int(time.hour)] + 1] += 1
						
# 					print cat, data['category']

# 		break
						




def createDataFrame(Files, target):
	for file in Files:
		print 'person', file
		table = [[0.0]*27 for _ in range(6)]
		# print table

		with open(file, 'r') as f:
			for line in f.readlines():
				data = json.loads(line)
				if data['category'] != []:
					top_cat = data['category'][0][0].split('/')[1]

					col_index = cat_order.index(top_cat)
					time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p')

					row_index = hr_index_bucket[int(time.hour)]
					# print row_index, col_index, table[row_index][col_index]
					table[row_index][col_index] += 1


		table = np.array(table)
		for index,row in enumerate(table):
			row_sum = np.sum(row)
			if row_sum > 0:
				table[index] = table[index]/row_sum
		# print table
		name = target+file.split('/')[1].split('_')[0]+'.csv'
		print name
		# np.savetxt(table, name)
		np.savetxt(name, table, delimiter=',')

		


# train_psiFiles = glob.glob("train-suicide/*.json")
# createDataFrame(train_psiFiles, 'bucketizedData/train-suicide/')

# train_nsiFiles = glob.glob("train-non-suicide/*.json")
# createDataFrame(train_nsiFiles, 'bucketizedData/train-non-suicide/')

# train_lsFiles = glob.glob("train-low-self-esteem/*.json")
# createDataFrame(train_lsFiles, 'bucketizedData/train-low-self-esteem/')

# train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")
# createDataFrame(train_nlsFiles, 'bucketizedData/train-not-low-self-esteem/')