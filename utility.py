import numpy as np
from sklearn.model_selection import LeaveOneOut
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# link to vader
# https://github.com/cjhutto/vaderSentiment

def sentiment_scores(sentence): 
	'''
	Given a sentence, this function returns a dictionary
	with the positive, negative, neutral and compound 
	sentiment. For example it returns a dic like this:

	{
		'pos': 0.487, 
		'compound': 0.431, 
		'neu': 0.513, 
		'neg': 0.0
	}
	'''
	# Create a SentimentIntensityAnalyzer object. 
	sid_obj = SentimentIntensityAnalyzer() 

	# polarity_scores method of SentimentIntensityAnalyzer 
	# oject gives a sentiment dictionary. 
	# which contains pos, neg, neu, and compound scores. 
	sentiment_dict = sid_obj.polarity_scores(sentence)
	return sentiment_dict


def get_data_for_leave_one_out(X, y):
	'''
	X is the feature matrix
	y is the corresponding label
	'''
	loo = LeaveOneOut()
	loo.get_n_splits(X)
	print(loo.get_n_splits(X))
	for train_index, test_index in loo.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		yield X_train, y_train, X_test, y_test


def plot_roc_auc_for_each_fold()

if __name__ == '__main__':
	X = np.array([[1, 2], [3, 4], [5,6],[7,8]])
	y = np.array([1, 0, 1, 0])


	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	for X_train, y_train, X_test, y_test in get_data_for_leave_one_out(X, y):
		print('Do trianing on the x train, y train and test on the x test and y_test')
		print(X_train.shape, y_train.shape)
		# ******************************************
		# prediction probabilities on test set
		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
		# ******************************************


		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		plt.plot(
			fpr, 
			tpr, 
			lw=1, 
			alpha=0.3,
			label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
		)

    	i += 1
	plt.plot(
		[0, 1],
		[0, 1],
		linestyle='--', 
		lw=2, 
		color='r',
		label='Chance', 
		alpha=.8
	)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(
		mean_fpr, 
		mean_tpr, 
		color='b',
		label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		lw=2, alpha=.8
	)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(
		mean_fpr,
		tprs_lower,
		tprs_upper,
		color='grey',
		alpha=.2,
		label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()