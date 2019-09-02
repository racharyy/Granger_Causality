import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
import theano
import theano.tensor as tt
from scipy import stats
import pymc3 as pm
from helper import *
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from graphical_model import *



data = load_pickle('data_for_graphical_model.pkl')
good_param_set = load_pickle('../Plots/good_set.pkl')

print(len(good_param_set))
N= len(data['se'])
median_f1_ar, mean_f1_ar = [], []
for conf in good_param_set:

	train_index = conf['train_index']
	prob_cand = set(train_index)
	train_data_dict, test_data_dict = {}, {}
	for key in data:
	    train_data_dict[key] = []
	for key in data:
	    test_data_dict[key] = []
	for ind in range(N):
	    if ind in train_index:    
	        for key in data:
	            train_data_dict[key].append(data[key][ind])
	    else:
	        for key in data:
	            test_data_dict[key].append(data[key][ind])
	for key in data:
	    train_data_dict[key] = np.array(train_data_dict[key])
	    test_data_dict[key] = np.array(test_data_dict[key])



	trace,median_f1,mean_f1 = train_and_test(train_data_dict,test_data_dict,12,)
	median_f1_ar.append(median_f1)
	mean_f1_ar.append(mean_f1)
	print(median_f1,mean_f1)


print(np.mean(np.array(median_f1_ar)),' : mean of the medians')
print(np.mean(np.array(mean_f1_ar)),' : mean of the means')




