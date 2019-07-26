import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]


def detectTrend(Files, ylabel, DELTA, which):
	ylabel = ylabel
	DELTA = DELTA

	# ylabel = 'Week'
	# DELTA = 8

	# ylabel = 'Month'
	# DELTA = 31
	user_cat_vectors_mapping = {}

	for file in Files:
		name = file.split('_')[0].split('/')[-1]
		with open(file, 'r') as f:


			i = 0
			start = None
			cat_vectors = []

			for line in f.readlines():
				data = json.loads(line)
				if data['category'] != []:
					top_cat = data['category'][0][0].split('/')[1]

					time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p') 

					if i == 0:
						start = time
						vector = [0.0]*len(cat_order)

					if i>0:
						delta = start - time

						if delta.days<DELTA:
							vector[cat_order.index(top_cat)] +=1

						else:
							cat_vectors.append(vector)
							start = time
							vector = [0.0]*len(cat_order)



					i+=1

		cat_vectors = np.array(cat_vectors)

		for index,row in enumerate(cat_vectors):
			row_sum = np.sum(row)
			if row_sum != 0.0:
				cat_vectors[index] = cat_vectors[index]/row_sum


		user_cat_vectors_mapping[name] = cat_vectors


	for name, catVecs in user_cat_vectors_mapping.iteritems():
		
		autocorrelation =[]
		for catIndex in range(len(cat_order)):
			catSig = catVecs[:,catIndex] - np.mean(catVecs[:,catIndex])
			for shift in range(len(catSig)):
				if shift == 0:
					correlation = np.corrcoef(catSig, catSig)[0,1]
				else:
					correlation = np.corrcoef(catSig[:-shift], catSig[shift:])[0,1]
				autocorrelation.append(correlation)

			print autocorrelation
			plt.plot(range(len(autocorrelation)), autocorrelation,'.r-', linewidth = 2,)
			plt.xlabel("Lag")
			plt.ylabel("Autocorrelation")
			plt.ylim(-1,1)
			plt.grid(True)
			# plt.xticks(np.arange(0, len(catSig), 1.0))
			plt.title("Category")
			plt.show()

			break
		print 

		



	# autocorrelation = []

	# for shift in range(20):
	# 	if shift == 0:
	# 		correlation = np.corrcoef(cluster0_averaged_pos, cluster0_averaged_pos)[0,1]
	# 	else:
	# 		correlation = np.corrcoef(cluster0_averaged_pos[:-shift], cluster0_averaged_pos[shift:])[0,1]
	# 	autocorrelation.append(correlation)

	# print autocorrelation
	# plt.plot(range(len(autocorrelation)), autocorrelation,'.r-', linewidth = 2,)
	# plt.xlabel("Tweet Lag")
	# plt.ylabel("Autocorrelation")
	# plt.ylim(-1,1)
	# plt.grid(True)
	# plt.xticks(np.arange(0, 21, 1.0))
	# plt.title("Cluster0: Periodicity of tweet with high positive sentiment")
	# plt.show()

train_psiFiles = glob.glob("train-low-self-esteem/*.json")
detectTrend(train_psiFiles, 'Month', 31, 'psi')