from driver import *
from config import tasks, methods, features, lambdapriors, catpriors
import sys
from helper import *
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')


ls_compound, nls_compound = load_pickle("../Boyu/compound_vectors_self_esteem.pkl")
psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
data = (ls_compound, nls_compound, psi_compound, npsi_compound)
print("Data Load done")

config_list = [{'task':0,'method':0,'features':1,'lambdapriors':0,'catpriors':1,'sparsity_hyperparam':0.02},
]




for config in config_list:

	driver = worker(data,config)
	cr,cm = driver.train_and_test()

	print("Task: ",tasks[config["task"]], " Method: ", methods[config["method"]], " Features: ", features[config["features"]])

	print(cr)
	print(cm)

	print("==================")
