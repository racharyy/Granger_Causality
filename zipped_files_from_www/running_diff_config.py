from driver import *
from config import tasks, methods, features, lambdapriors, catpriors
import sys
from helper import *
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')

orig_stdout = sys.stdout
f = open('../Plots/out.txt', 'w')

ls_compound, nls_compound = load_pickle("../Boyu/compound_vectors_self_esteem.pkl")
psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
npsi_compound = npsi_compound[:37]+npsi_compound[38:]
data = (ls_compound, nls_compound, psi_compound, npsi_compound)
print("Data Load done")

with open('../Plots/out.txt', 'a') as f:

	config_list = []

	for i in [0]:#range(len(tasks)):
		for j in [1]:#range(len(methods)):
			for k in range(len(features)):
				for l in [0]:#range(len(lambdapriors)):
					for m in [0]:#range(len(catpriors)):

						config_dic = {'task':i, 'method':j, 'features':k, 'lambdapriors':l, 'catpriors':m, 'sparsity_hyperparam':0.02}
						config_list.append(config_dic)
	#print(config_list)



	driver = worker(data,{})
	driver.data_split()

	for ind,config in enumerate(config_list):
		driver.config = config
		print(ind)
		if config["features"]==0:
			print("Task: ",tasks[config["task"]], "||||||||| Method: ", methods[config["method"]], "||||||||| Features: ", features[config["features"]], "||||||||| lambdapriors: ", lambdapriors[config["lambdapriors"]],file=f)
		elif config["features"]==1:
			print("Task: ",tasks[config["task"]], "||||||||| Method: ", methods[config["method"]], "||||||||| Features: ", features[config["features"]], "||||||||| catpriors: ", catpriors[config["catpriors"]],file=f)
		else:
			print("Task: ",tasks[config["task"]], "||||||||| Method: ", methods[config["method"]], "||||||||| Features: ", features[config["features"]], "||||||||| lambdapriors: ", lambdapriors[config["lambdapriors"]], "||||||||| catpriors: ", catpriors[config["catpriors"]],file=f)


		
		cr,cm = driver.train_and_test()

		
		print(cr,file=f)
		#print(cm)

		print("==================",file=f)
