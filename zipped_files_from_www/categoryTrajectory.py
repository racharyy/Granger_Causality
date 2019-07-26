import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]


def plotCategoryTrajectory(Files, ylabel, DELTA,where):

	ylabel = ylabel
	DELTA = DELTA

	# ylabel = 'Week'
	# DELTA = 8

	# ylabel = 'Month'
	# DELTA = 31

	for file in Files:
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


		if cat_vectors.shape[0]!=0:

			sns.heatmap(cat_vectors,cmap="YlGnBu")
			plt.title(ylabel+' Search Category Trajectory (by %)')
			plt.ylabel(ylabel)
			plt.xticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=90,fontsize=8)
			plt.yticks(fontsize=6)
			name = file.split('_')[0].split('/')[-1]
			plt.subplots_adjust(bottom=0.3)
			plt.tight_layout()
			
			title = name+'_'+ylabel
			print 'saving' , title
			plt.savefig('categories/trajectory/'+where+'/'+title+".png")
			# plt.show()
			plt.close()
	
# train_lsFiles = glob.glob("train-low-self-esteem/*.json")

# plotCategoryTrajectory(train_lsFiles, 'Week', 8, 'ls')
# plotCategoryTrajectory(train_lsFiles, 'Month', 31, 'ls')
# plotCategoryTrajectory(train_lsFiles, 'Daily', 1, 'ls')

# train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")
# plotCategoryTrajectory(train_nlsFiles, 'Week', 8, 'nls')
# plotCategoryTrajectory(train_nlsFiles, 'Month', 31, 'nls')
# plotCategoryTrajectory(train_nlsFiles, 'Daily', 1, 'nls')


# train_psiFiles = glob.glob("train-suicide/*.json")
# plotCategoryTrajectory(train_psiFiles, 'Week', 8, 'psi')
# plotCategoryTrajectory(train_psiFiles, 'Month', 31, 'psi')
# plotCategoryTrajectory(train_psiFiles, 'Daily', 1, 'psi')


# train_nsiFiles = glob.glob("train-non-suicide/*.json")
# plotCategoryTrajectory(train_nsiFiles, 'Week', 8, 'nsi')
# plotCategoryTrajectory(train_nsiFiles, 'Month', 31, 'nsi')
# plotCategoryTrajectory(train_nsiFiles, 'Daily', 1, 'nsi')