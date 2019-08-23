from io import open
import pickle
import sys
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')

from posterior import *
from joint_posterior import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from copy import copy
#from config import config
from helper import *
from config import *
import scikitplot as skplt


class light_worker(object):
	"""docstring for light_worker"""
	def __init__(self, data,config):
		super(light_worker, self).__init__()
		self.data = data
		self.config = config


	def data_split(self, split_ratio = 3.0/4):
		
		ls_compound, nls_compound, psi_compound, npsi_compound = self.data
		#psi_pretrained, npsi_pretrained, ls_pretrained,nls_pretrained  = load_pickle('../Boyu/best_representation.pkl')
		
		
		lsnls_train_feature, lsnls_train_label, lsnls_test_feature, lsnls_test_label, lsnls_train_user, lsnls_test_user = random_split(ls_compound,nls_compound,split_ratio)
		psinpsi_train_feature, psinpsi_train_label, psinpsi_test_feature, psinpsi_test_label, psinpsi_train_user, psinpsi_test_user = random_split(psi_compound,npsi_compound,split_ratio)
		
		lsnls_train_lambda, lsnls_train_cat, lsnls_test_lambda, lsnls_test_cat = [], [], [], []	
		for user in lsnls_train_feature:
			lsnls_train_cat.append(user[:27])
			lsnls_train_lambda.append(user[27:])
		for user in lsnls_test_feature:
			lsnls_test_cat.append(user[:27])
			lsnls_test_lambda.append(user[27:])

		psinpsi_train_lambda, psinpsi_train_cat, psinpsi_test_lambda, psinpsi_test_cat = [], [], [], []	
		for user in psinpsi_train_feature:
			psinpsi_train_cat.append(user[:27])
			psinpsi_train_lambda.append(user[27:])
		for user in psinpsi_test_feature:
			psinpsi_test_cat.append(user[:27])
			psinpsi_test_lambda.append(user[27:])


		self.lsnls_data = (lsnls_train_feature,lsnls_train_lambda,lsnls_train_cat,lsnls_train_label,lsnls_test_feature,lsnls_test_lambda,lsnls_test_cat,lsnls_test_label)
		self.psinpsi_data = (psinpsi_train_feature,psinpsi_train_lambda,psinpsi_train_cat,psinpsi_train_label,psinpsi_test_feature,psinpsi_test_lambda,psinpsi_test_cat,psinpsi_test_label)
		self.lsnls_user = (lsnls_train_user, lsnls_test_user)
		self.psinpsi_user = (psinpsi_train_user, psinpsi_test_user)

		
		# for i in self.lsnls_data:
		# 	print(np.array(i).shape)
		# for i in self.psinpsi_data:
		# 	print(np.array(i).shape)	
		#return train_feature,train_lambda,train_cat,train_label,test_feature,test_lambda,test_cat,test_label


	def train_and_test(self):

		if self.config['task'] == 0:
			train_feature,train_lambda,train_cat,train_label,test_feature,test_lambda,test_cat,test_label = self.lsnls_data
			train_user, test_user = self.lsnls_user
		else:
			train_feature,train_lambda,train_cat,train_label,test_feature,test_lambda,test_cat,test_label = self.psinpsi_data
			train_user, test_user = self.psinpsi_user


		if self.config['method'] == 0:

			classifier = LogisticRegression(random_state=0, solver='liblinear')

		else:

			classifier = SVC(gamma='auto')

		if self.config['features'] == 0:

			clf = classifier.fit(train_lambda, train_label)
			y_hat = clf.predict(test_lambda)

		elif self.config['features'] == 1:

			clf = classifier.fit(train_cat, train_label)
			y_hat = clf.predict(test_cat)

		elif self.config['features'] == 2:

			clf = classifier.fit(train_feature, train_label)
			y_hat = clf.predict(test_feature)

		elif self.config['features'] == 3:

			psi, npsi,ls,nls,ls_lin, nls_lin,psi_lin,npsi_lin  = load_pickle('../Boyu/best_representation.pkl')
			psi_c, npsi_c,ls_c,nls_c,ls_lin_c, nls_lin_c,psi_lin_c,npsi_lin_c  = load_pickle('../Boyu/best_representation_1.pkl')

			if self.config['task'] == 0: 
				if self.config['method'] == 2: #or self.config['method'] == 1:
					train_lambda, train_label, test_lambda, test_label = extract_index(ls_lin,nls_lin,train_user,test_user)
				else:
					train_lambda, train_label, test_lambda, test_label = extract_index(ls,nls,train_user,test_user)
			else:
				if self.config['method'] == 2:# or self.config['method'] == 1:
					train_lambda, train_label, test_lambda, test_label = extract_index(psi_lin,npsi_lin,train_user,test_user)
				else:
					train_lambda, train_label, test_lambda, test_label = extract_index(psi,npsi,train_user,test_user)

			clf = classifier.fit(train_lambda, train_label)
			y_hat = clf.predict(test_lambda)


		cr = classification_report(test_label, y_hat)
		avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
					
		return float(avg_prec), float(avg_recal), float(avg_f1)

