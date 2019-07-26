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
all_cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]
ssd_cats = ['Autos & Vehicles', 'Business & Industrial', 'Food & Drink', 'Hobbies & Leisure', 'Travel']
# 23,2,11,7,20,4
# (2,4,7,11,20,23)
def do_svm(train_x, train_y, test_x, test_y, c=0.5, solver='linear'):
	from sklearn.model_selection import GridSearchCV
	# Set the parameters by cross-validation
	tuned_parameters = [
		#{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.05, 0.1, 0.5, 1, 10, 100, 1000]},
		{'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 1, 10, 100, 1000]}
	]

	scores = ['precision', 'recall']

	for score in scores:
		print "# Tuning hyper-parameters for %s" % score
		print "\n"

		clf = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring='%s_macro' % score)
		best_model = clf.fit(train_x, train_y)

		print "Best parameters set found on development set:"
		print "\n"
		print clf.best_params_


		print "Grid scores on development set:"
		print"\n"
		model_coef = best_model.best_estimator_.coef_[0]
		print model_coef
		feature_weights = np.argsort(model_coef)
		print feature_weights
		print 'number of features', len(feature_weights)


		print feature_weights[::-1]
		for index in feature_weights[::-1]:
			print all_cats[index], model_coef[index]








		print "Grid scores on development set:"
		print"\n"
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		# 	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		# print"\n"

		print "Detailed classification report:" 
		print"\n"

		print "The model is trained on the full development set."
		print "The scores are computed on the full evaluation set."
		print"\n"
		test_result = []
		y_true, y_pred = test_y, clf.predict(test_x)
		print classification_report(y_true, y_pred)
		print 'Model Accuracy:', accuracy_score(y_true, y_pred)
		for i in range(len(y_pred)):
			test_result.append((y_true[i], y_pred[i]))
			print 'true:', y_true[i], '\t pred:', y_pred[i]
		print "\n"





		return test_result
def do_logistic_regression(train_x, train_y, test_x, test_y):

	train_x = np.nan_to_num(train_x)
	test_x = np.nan_to_num(test_x)


	C = [0.5, 1, 5, 10, 150, 50, 200, 300, 500, 700, 1000]

	best_c = 0
	best_score = 0.0
	
	for c in C:
		# print '\t C:', c, 'Solver:liblinear', 'Regularization: l2'
		# print '\t - - - - - - - - - - - - - - - - - - - - - - - - - '
		clf = linear_model.LogisticRegression(
			C = c, 
			solver='newton-cg', 
			penalty='l2', 
			max_iter=10000000, 
			random_state=4
		)
		scores = cross_validation.cross_val_score(clf, train_x, train_y, cv=10)
		# print "\t Accuracy for validations: ", scores
		# print "\t Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

		if scores.mean() > best_score:
			best_score = scores.mean()
			best_c = c
	
	print '** Best C after cross validation:', best_c
	
	print 'CV and Fine tuning is done on Training Set!\n\n'
	print '====================================='
	print '\n ======== Starting to train ======== \n'
	clf = linear_model.LogisticRegression(C = best_c, solver='liblinear', penalty='l2', max_iter=100000, random_state=42, n_jobs=-1)
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

	print clf.coef_
	# model's the features in the decision function
	# feature_importnace = []
	# for idx,bool in enumerate(selected_features):
	# 	if bool == True:
	# 		feature_importnace.append((all_train[idx] , clf.coef_[0][idx]))


	# feature_importnace.sort(key=lambda x: x[1],reverse=True)
	# for i in feature_importnace:
	# 	print i
	return test_result


def load_psi_nsi_data():
	train_psi_data = np.loadtxt(
		"si_cat_search_percent_dist_by_users.csv",
		skiprows=1,
		usecols=range(1,14),
		dtype='float32', 
		delimiter=','
	)
	train_psi_data = train_psi_data.transpose()

	train_npsi_data = np.loadtxt(
		"nsi_cat_search_percent_dist_by_users.csv",
		skiprows=1,
		usecols=range(1,42),
		dtype='float32', 
		delimiter=','
	)
	train_npsi_data = train_npsi_data.transpose()
	print 'PSI train shape:',train_psi_data.shape
	print 'nPSI train shape:',train_npsi_data.shape



	test_psi_data = np.loadtxt(
		"si_cat_search_percent_dist_by_users_validation.csv",
		skiprows=1,
		usecols=range(1,6),
		dtype='float32', 
		delimiter=','
	)
	test_psi_data = test_psi_data.transpose()

	test_npsi_data = np.loadtxt(
		"nsi_cat_search_percent_dist_by_users_validation.csv",
		skiprows=1,
		usecols=range(1,14),
		dtype='float32', 
		delimiter=','
	)
	test_npsi_data = test_npsi_data.transpose()
	print 'PSI test shape:',test_psi_data.shape
	print 'nPSI test shape:',test_npsi_data.shape
	return train_psi_data, train_npsi_data, test_psi_data,test_npsi_data


ssd_cols = [22,1,10,6,19,3]
ALL = True
train_psi_data, train_npsi_data, test_psi_data,test_npsi_data = load_psi_nsi_data()

psi_nsi_class_label = [1]*len(train_psi_data)
psi_nsi_class_label = psi_nsi_class_label + [0] * len(train_npsi_data)

psi_nsi_y = np.array(psi_nsi_class_label)
psi_npsi_X = np.concatenate((train_psi_data,train_npsi_data), axis=0)

if ALL == False:
	psi_npsi_X = psi_npsi_X[:,ssd_cols]

test_psi_npsi_class_label = [1]*len(test_psi_data)
test_psi_npsi_class_label = test_psi_npsi_class_label + [0] * len(test_npsi_data)

test_psi_npsi_y = np.array(test_psi_npsi_class_label)
test_psi_npsi_X = np.concatenate((test_psi_data,test_npsi_data), axis=0)
if ALL == False:
	test_psi_npsi_X = test_psi_npsi_X[:,ssd_cols]

svm_res = do_svm(psi_npsi_X, psi_nsi_y, test_psi_npsi_X, test_psi_npsi_y, c=0.5, solver='liblinear')