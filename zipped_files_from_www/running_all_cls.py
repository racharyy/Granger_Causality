from all_cls import *
from all_cls_config import tasks, methods, classifiers, features
assert len(methods) == len (classifiers)

import sys
from helper import *
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')

orig_stdout = sys.stdout
f = open('../Plots/out_all_cls.txt', 'w')

ls_compound, nls_compound = load_pickle("../Boyu/compound_vectors_self_esteem.pkl")
psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
# npsi_compound = npsi_compound[:37]+npsi_compound[38:]
data = (ls_compound, nls_compound, psi_compound, npsi_compound)
print("Data Load done")

num_trials = 100
config_list = []

avg_prec_ar, avg_recal_ar, avg_f1_ar,avg_aic_ar = np.zeros((len(tasks),len(methods),len(features))), np.zeros((len(tasks),len(methods),len(features))), np.zeros((len(tasks),len(methods),len(features))), np.zeros((len(tasks),len(methods),len(features)))

with open('../Plots/out_all_cls.txt', 'a') as f:
	for i in range(len(tasks)):
		for j in range(len(methods)):
			for k in range(len(features)):
				config_dic = {'task':i, 'method':j, 'classifier':classifiers[j], 'features':k}
				config_list.append(config_dic)

	# print(config_list)
	for trials in range(num_trials):

		driver = light_worker(data,{})
		driver.data_split()

		if trials % 10==0:
			print(trials)

		for config in config_list:
			
			driver.config = config
			avg_prec, avg_recal, avg_f1,aic = driver.train_and_test()
			task, method, feature = config['task'], config['method'], config['features']

			#print(avg_prec,task,method,features)
			avg_prec_ar[task,method,feature] = avg_prec_ar[task,method,feature]+avg_prec
			avg_recal_ar[task][method][feature] = avg_recal_ar[task][method][feature]+avg_recal
			avg_f1_ar[task][method][feature] = avg_f1_ar[task][method][feature]+avg_f1
			avg_aic_ar[task][method][feature] = avg_aic_ar[task][method][feature]+aic

	avg_prec_ar, avg_recal_ar,avg_f1_ar, avg_aic_ar = (1.0/num_trials)*avg_prec_ar, (1.0/num_trials)*avg_recal_ar, (1.0/num_trials)*avg_f1_ar, (1.0/num_trials)*avg_aic_ar

	print("--------------   Precission Results   --------------",file=f)

	for i in range(len(tasks)):
		for j in range(len(methods)):
			for k in range(len(features)):
				print("For Task: ",tasks[i], " with method: ",methods[j], " with feature: " ,features[k]," Average precesion is: ", avg_prec_ar[i,j,k],file=f )

	print("--------------   Recall Results   --------------",file=f)

	for i in range(len(tasks)):
		for j in range(len(methods)):
			for k in range(len(features)):
				print("For Task: ",tasks[i], " with method: ",methods[j], " with feature: " ,features[k]," Average recall is: ", avg_recal_ar[i,j,k],file=f )


	print("--------------   F1 Score Results   --------------",file=f)

	for i in range(len(tasks)):
		for j in range(len(methods)):
			for k in range(len(features)):
				print("For Task: ",tasks[i], " with method: ",methods[j], " with feature: " ,features[k]," Average F1 Score is: ", avg_f1_ar[i,j,k],file=f )

	print("--------------   AIC Score Results   --------------",file=f)

	for i in range(len(tasks)):
		for j in range(len(methods)):
			for k in range(len(features)):
				print("For Task: ",tasks[i], " with method: ",methods[j], " with feature: " ,features[k]," Average AIC Score is: ", avg_aic_ar[i,j,k],file=f )

f.close()
