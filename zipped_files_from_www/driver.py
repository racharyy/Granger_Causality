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

class worker(object):
	"""docstring for worker"""
	def __init__(self, data,config):
		super(worker, self).__init__()
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

		#load covariance matrix and use if needed
		cov_W = np.loadtxt("similarity_matrix.csv", dtype='float32', delimiter=',')
		mu_W = np.zeros(len(cats))

		# feature 0: lambda
		if self.config['features'] == 0 and self.config['method'] == 0:

			bmodel = Bayes_model(np.array(train_lambda), train_label, cov_W, mu_W, self.config['sparsity_hyperparam'])
			if self.config['lambdapriors'] == 0:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = True)
			else:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = False)
			
			best_f1 = 0
			y_best = 0
			f1s =[]
			aucs=np.zeros(len(samples_w))
			for i in range(len(samples_w)):

				w = samples_w[i]
				sig = samples_sigma[i]

				p = sigmoid(np.dot(test_lambda, w) + sig*np.random.normal())

				y_hat = [np.random.binomial(1, j) for j in p]

				cr = classification_report(test_label, y_hat)
				avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
				# print(cr.split('avg / total')[-1].strip().split())
				# print(avg_prec, avg_recal,avg_f1)
				# assert(False)
				f1s.append(float(avg_f1))
				if best_f1<float(avg_f1):
					best_f1 = copy(float(avg_f1))
					y_best=copy(y_hat)
				#print(p)
				#print(len(p),len(test_label))
				# probas=[[1-i,i] for i in p]
				# skplt.metrics.plot_roc_curve(test_label, probas)
				# plt.show()
				# break

				fpr, tpr, thresholds = roc_curve(test_label, p)
				#print(y_test,  probas_[:, 1])
				# tprs.append(interp(mean_fpr, fpr, tpr))
				# tprs[-1][0] = 0.0
				aucs[i]=auc(fpr, tpr)
			mean_auc = np.mean(aucs)
			mean_f1 = np.median(np.array(f1s))
			print(mean_f1,features[self.config['features']])
			y_hat = copy(y_best)
			#print(best_f1)

			exp_name = tasks[self.config['task']]+"_"+features[self.config['features']]+"_"+lambdapriors[self.config['lambdapriors']]
			plot_name = '_avgf1_hist.png'
			plt.hist(f1s,bins=30)
			plt.title('f1s')
			plt.savefig('../Plots/'+exp_name +plot_name)
			plt.close()


		elif self.config['features'] == 0 and self.config['method'] == 1:
			#print(train_label)
			clf = LogisticRegression(random_state=0, solver='liblinear').fit(train_lambda,train_label)
			y_hat = clf.predict(test_lambda)
			# clf.predict_proba(test_lambda)
			#print(clf.predict_proba(test_lambda))
			# fpr, tpr, thresholds = roc_curve(test_label, p)
			# 	#print(y_test,  probas_[:, 1])
			# 	# tprs.append(interp(mean_fpr, fpr, tpr))
			# 	# tprs[-1][0] = 0.0
			# 	aucs[i]=auc(fpr, tpr)
			# mean_auc = np.mean(aucs)
			# print(clf.score(test_lambda,test_label))

		elif self.config['features'] == 0 and self.config['method'] == 2:

			clf = SVC(gamma='auto').fit(train_lambda, train_label)
			y_hat = clf.predict(test_lambda)

		# features 1: category
		elif self.config['features'] == 1 and self.config['method'] == 0:

			bmodel = Bayes_model(np.array(train_cat), train_label, cov_W, mu_W, self.config['sparsity_hyperparam'])
			if self.config['catpriors'] ==0:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = True)
			else:
				samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = False)
			

			best_f1 = 0
			y_best = 0
			f1s =[]
			aucs=np.zeros(len(samples_w))
			for i in range(len(samples_w)):

				w = samples_w[i]
				sig = samples_sigma[i]

				p = sigmoid(np.dot(test_lambda, w) + sig*np.random.normal())
				y_hat = [np.random.binomial(1, j) for j in p]

				cr = classification_report(test_label, y_hat)
				avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
				f1s.append(float(avg_f1))
				if best_f1<float(avg_f1):
					best_f1 = copy(float(avg_f1))
					y_best=copy(y_hat)

				fpr, tpr, thresholds = roc_curve(test_label, p)
				#print(y_test,  probas_[:, 1])
				# tprs.append(interp(mean_fpr, fpr, tpr))
				# tprs[-1][0] = 0.0
				aucs[i]=auc(fpr, tpr)
			mean_auc = np.mean(aucs)
			mean_f1 = np.median(np.array(f1s))
			print(mean_f1,features[self.config['features']])
			y_hat = copy(y_best)
			exp_name = tasks[self.config['task']]+"_"+features[self.config['features']]+"_"+catpriors[self.config['catpriors']]
			plot_name = '_avgf1_hist.png'
			plt.hist(f1s,bins=30)
			plt.title('f1s')
			plt.savefig('../Plots/'+exp_name +plot_name)
			plt.close()

		elif self.config['features'] == 1 and self.config['method'] == 1:

			clf = LogisticRegression(random_state=0, solver='liblinear').fit(train_cat,train_label)
			#print(clf.predict_proba(test_cat))
			y_hat=clf.predict(test_cat)

		elif self.config['features'] == 1 and self.config['method'] == 2:

			clf = SVC(gamma='auto').fit(train_cat,train_label)
			y_hat = clf.predict(test_cat)

		elif self.config['features'] == 2 and self.config['method'] == 0:

			bmodel = Bayes_model_joint((np.array(train_lambda),np.array(train_cat)),train_label,cov_W, mu_W,self.config)
			samples_w1,samples_w2, samples_sigma = bmodel.metropolis_hastings()

			best_f1 = 0
			y_best = []
			f1s =[]
			aucs=np.zeros(len(samples_w1))
			for i in range(len(samples_w1)):

				w1 = samples_w1[i]
				w2 = samples_w2[i]
				sig = samples_sigma[i]

				p = sigmoid(np.dot(test_lambda, w1)+ np.dot(test_cat, w2)+ sig*np.random.normal())
				y_hat = [np.random.binomial(1, j) for j in p]

				cr = classification_report(test_label, y_hat)
				avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
				
				f1s.append(float(avg_f1))
				if best_f1<float(avg_f1):
					best_f1 = copy(float(avg_f1))
					y_best=copy(y_hat)
				fpr, tpr, thresholds = roc_curve(test_label, p)
				#print(y_test,  probas_[:, 1])
				# tprs.append(interp(mean_fpr, fpr, tpr))
				# tprs[-1][0] = 0.0
				aucs[i]=auc(fpr, tpr)
			mean_auc = np.mean(aucs)
			mean_f1 = np.median(np.array(f1s))
			print(mean_f1,features[self.config['features']])
			y_hat = copy(y_best)
			exp_name = tasks[self.config['task']]+"_"+features[self.config['features']]+"_"+lambdapriors[self.config['lambdapriors']]+"_"+catpriors[self.config['catpriors']]
			plot_name = '_avgf1_hist.png'
			plt.hist(f1s,bins=30)
			plt.title('f1s')
			plt.savefig('../Plots/'+exp_name +plot_name)
			plt.close()

		elif self.config['features'] == 2 and self.config['method'] == 1:

			clf = LogisticRegression(random_state=0, solver='liblinear').fit(train_feature,train_label)
			#print(clf.predict_proba(test_feature))
			y_hat=clf.predict(test_feature)

		elif self.config['features'] == 2 and self.config['method'] == 2:

			clf = SVC(gamma='auto').fit(train_feature,train_label)
			y_hat = clf.predict(test_feature)


		elif self.config['features'] == 3:

			psi, npsi,ls,nls,ls_lin, nls_lin,psi_lin,npsi_lin  = load_pickle('../Boyu/best_representation.pkl')
			psi_c, npsi_c,ls_c,nls_c,ls_lin_c, nls_lin_c,psi_lin_c,npsi_lin_c  = load_pickle('../Boyu/best_representation_1.pkl')

			if self.config['task'] == 0: 
				if self.config['method'] == 0: #or self.config['method'] == 1:
					train_lambda, train_label, test_lambda, test_label = extract_index(ls_lin,nls_lin,train_user,test_user)
				else:
					train_lambda, train_label, test_lambda, test_label = extract_index(ls,nls,train_user,test_user)
			else:
				if self.config['method'] == 0:# or self.config['method'] == 1:
					train_lambda, train_label, test_lambda, test_label = extract_index(psi_lin,npsi_lin,train_user,test_user)
				else:
					train_lambda, train_label, test_lambda, test_label = extract_index(psi,npsi,train_user,test_user)
			#print(len(train_lambda),len(test_lambda))
			#train_lambda, train_label, test_lambda, test_label = random_split(psi,npsi,split_ratio)
			num_hidden_features = len(train_lambda[0])
			mu_W = np.zeros(num_hidden_features)
			cov_W = np.eye(num_hidden_features)

			if self.config['method'] == 0:
				bmodel = Bayes_model(np.array(train_lambda), train_label, cov_W, mu_W, self.config['sparsity_hyperparam'])
				if self.config['lambdapriors'] == 0:
					samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = True)
				else:
					samples_w, samples_sigma = bmodel.metropolis_hastings(sparsity_Flag = False)
				
				best_f1 = 0
				y_best = 0
				f1s =[]
				aucs=np.zeros(len(samples_w))
				for i in range(len(samples_w)):

					w = samples_w[i]
					sig = samples_sigma[i]

					p = sigmoid(np.dot(test_lambda, w) + sig*np.random.normal())
					y_hat = [np.random.binomial(1, j) for j in p]

					cr = classification_report(test_label, y_hat)
					avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
					# print(cr.split('avg / total')[-1].strip().split())
					# print(avg_prec, avg_recal,avg_f1)
					# assert(False)
					f1s.append(float(avg_f1))
					if best_f1<float(avg_f1):
						best_f1 = copy(float(avg_f1))
						y_best=copy(y_hat)
					fpr, tpr, thresholds = roc_curve(test_label, p)
				#print(y_test,  probas_[:, 1])
				# tprs.append(interp(mean_fpr, fpr, tpr))
				# tprs[-1][0] = 0.0
					aucs[i]=auc(fpr, tpr)
				mean_auc = np.mean(aucs)
				mean_f1 = np.median(np.array(f1s))
				print(mean_f1,features[self.config['features']])
				y_hat = copy(y_best)
				#print(best_f1)

				exp_name = tasks[self.config['task']]+"_"+features[self.config['features']]+"_"+lambdapriors[self.config['lambdapriors']]
				plot_name = '_avgf1_hist.png'
				plt.hist(f1s,bins=30)
				plt.title('f1s')
				plt.savefig('../Plots/'+exp_name +plot_name)
				plt.close()

			elif self.config['method'] == 1:
				#print(len(train_lambda[0]))
				# print("=========")
				# print(train_user)
				# print("---------")
				# print(test_user)
				clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=3000).fit(train_lambda,train_label)
				#print(clf.predict_proba(test_lambda))
				y_hat=clf.predict(test_lambda)

			elif self.config['method'] == 2:

				clf = SVC(gamma='auto',kernel='sigmoid').fit(train_lambda,train_label)
				y_hat = clf.predict(test_lambda)

		cr = classification_report(test_label, y_hat)
		#print(cr)
		cm = confusion_matrix(test_label, y_hat)
		return cr,cm




