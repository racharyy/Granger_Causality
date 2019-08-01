from io import open
import pickle
import sys
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')

from posterior import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#from config import config
from helper import random_split
from config import cats

class worker(object):
	"""docstring for worker"""
	def __init__(self, data,config):
		super(worker, self).__init__()
		self.data = data
		self.config = config
		

	def train_and_test(self, split_ratio = 2/3):

		ls_compound, nls_compound, psi_compound, npsi_compound = self.data
		
		if self.config['task'] == 0:
			train_feature, train_label, test_feature, test_label = random_split(ls_compound,nls_compound,split_ratio)
		else:
			train_feature, train_label, test_feature, test_label = random_split(psi_compound,npsi_compound,split_ratio)
		
		train_lambda, train_cat, test_lambda, test_cat = [], [], [], []	
		for user in train_feature:
			train_cat.append(user[:27])
			train_lambda.append(user[27:])
		for user in test_feature:
			test_cat.append(user[:27])
			test_lambda.append(user[27:])

		#load covariance matrix and use if needed
		cov_W = np.loadtxt("similarity_matrix.csv", dtype='float32', delimiter=',')
		mu_W = np.zeros(len(cats))

		# feature 0: lambda
		if self.config['features'] == 0 and self.config['method'] == 0:

			bmodel = Bayes_model(np.array(train_lambda), train_label, cov_W, mu_W, self.config['sparsity_hyperparam'])
			if self.config['lambdapriors'] == 0:
				samples_w, samples_sigma = bmodel.metropolis_hastings()
			else:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = False)
			w = np.mean(samples_w, axis=0)
			sig = np.mean(samples_sigma)

			p = sigmoid(np.dot(test_lambda, w) + sig*np.random.normal())
			y_hat = []
			for i in p:
				if i <= 0.5:
					y_hat.append(1)
				else:
					y_hat.append(0)


		elif self.config['features'] == 0 and self.config['method'] == 1:

			clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_lambda,train_label)
			# print(clf.score(train_lambda,train_label))
			y_hat = clf.predict(test_lambda)
			# print(clf.score(test_lambda,test_label))

		elif self.config['features'] == 0 and self.config['method'] == 2:

			clf = SVC(gamma='auto').fit(train_lambda, train_label)
			y_hat = clf.predict(test_lambda)

		# features 1: category
		elif self.config['features'] == 1 and self.config['method'] == 0:

			bmodel = Bayes_model(np.array(train_cat), train_label, cov_W, mu_W, self.config['sparsity_hyperparam'])
			if self.config['catpriors'] ==0:
				samples_w, samples_sigma = bmodel.metropolis_hastings()
			else:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = False)
			w = np.mean(samples_w, axis=0)
			sig = np.mean(samples_sigma)

			p = sigmoid(np.dot(test_cat, w) + sig*np.random.normal())
			y_hat = []
			for i in p:
				if i <= 0.5:
					y_hat.append(1)
				else:
					y_hat.append(0)

		elif self.config['features'] == 1 and self.config['method'] == 1:

			clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_cat,train_label)
			y_hat=clf.predict(test_cat)

		elif self.config['features'] == 1 and self.config['method'] == 2:

			clf = SVC(gamma='auto').fit(train_cat,train_label)
			y_hat = clf.predict(test_cat)

		elif self.config['features'] == 2 and self.config['method'] == 0:

			raise NotImplementedError

		elif self.config['features'] == 2 and self.config['method'] == 1:

			clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_feature,train_label)
			y_hat=clf.predict(test_lambda)

		elif self.config['features'] == 2 and self.config['method'] == 2:

			raise NotImplementedError

		cr = classification_report(test_label, y_hat)
		cm = confusion_matrix(test_label, y_hat)
		return cr,cm




