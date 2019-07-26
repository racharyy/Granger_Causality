import csv
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold

from sklearn.externals import joblib
from sklearn import linear_model, svm
from scipy import interp
from sklearn.feature_selection import VarianceThreshold



import sklearn, numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score



def load_ls_nls_data():
	train_ls_data = np.loadtxt(
		"ls_cat_search_percent_dist_by_users.csv",
		skiprows=1,
		usecols=range(1,28),
		dtype='float32', 
		delimiter=','
	)
	train_ls_data = train_ls_data.transpose()

	# 'ls_cat_search_percent_dist_by_users.csv'
	# 'nls_cat_search_percent_dist_by_users.csv'

	train_nls_data = np.loadtxt(
		"nls_cat_search_percent_dist_by_users.csv",
		skiprows=1,
		usecols=range(1,30),
		dtype='float32', 
		delimiter=','
	)
	train_nls_data = train_nls_data.transpose()

	print 'LS train shape:',train_ls_data.shape
	print 'NLS train shape:',train_nls_data.shape


	test_ls_data = np.loadtxt(
		"ls_cat_search_percent_dist_by_users_validation.csv",
		skiprows=1,
		usecols=range(1,20),
		dtype='float32', 
		delimiter=','
	)
	test_ls_data = test_ls_data.transpose()



	test_nls_data = np.loadtxt(
		"nls_cat_search_percent_dist_by_users_validation.csv",
		skiprows=1,
		usecols=range(1,5),
		dtype='float32', 
		delimiter=','
	)
	test_nls_data = test_nls_data.transpose()
	print '-----'
	print 'LS test shape:',test_ls_data.shape
	print 'NLS test shape:',test_nls_data.shape
	return train_ls_data, train_nls_data, test_ls_data, test_nls_data




train_ls_data, train_nls_data, test_ls_data, test_nls_data = load_ls_nls_data()


###################################################
# Model low self esteem
ls_nls_class_label = [1]*len(train_ls_data)
ls_nls_class_label = ls_nls_class_label + [0] * len(train_nls_data)


ls_nls_X = np.concatenate((train_ls_data,train_nls_data), axis=0)
ls_nls_y = np.array(ls_nls_class_label)

print 'ls-nls X', ls_nls_X.shape
print 'ls-nls y', ls_nls_y.shape
# np.save('ls_nls_x', ls_nls_X)
# np.save('ls_nls_y', ls_nls_y)



test_ls_nls_class_label = [1]*len(test_ls_data)
test_ls_nls_class_label = test_ls_nls_class_label + [0] * len(test_nls_data)


test_ls_nls_X = np.concatenate((test_ls_data,test_nls_data), axis=0)
test_ls_nls_y = np.array(test_ls_nls_class_label)

print 'TEST: ls-nls X', test_ls_nls_X.shape
print 'TEST: ls-nls y', test_ls_nls_y.shape






train_x, train_y, test_x, test_y = ls_nls_X, ls_nls_y, test_ls_nls_X, test_ls_nls_y

clf = linear_model.LogisticRegression()
# clf = linear_model.LogisticRegression(C = 0.5, solver='liblinear', penalty='l2', max_iter=100000, random_state=42, n_jobs=-1)
clf.fit(train_x, train_y)
print 'training score:'
print clf.score
print '----------------'
print 'Saving model:'
# joblib.dump(clf, 'log_reg_model.pkl')
print 'Testing:....'
y_hat = clf.predict(test_x)
test_result = []

print 'Testing report: '
for i in range(len(y_hat)):
	print 'true:',test_y[i], '\t pred:', y_hat[i]
	test_result.append((test_y[i],y_hat[i]))
print classification_report(test_y, y_hat)
print 'Accuracy on Test set:',accuracy_score(test_y, y_hat)
print '====================================='
