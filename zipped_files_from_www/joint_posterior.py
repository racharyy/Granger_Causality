import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp 
from auxilary import *
from numpy.linalg import inv, det
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score

class Bayes_model_joint(object):
	"""docstring for Bayes_model"""
	def __init__(self,data,label,cov_W, mu_W,config):
		super(Bayes_model_joint, self).__init__()
		
		self.data = data
		self.label = label
		self.num_cat = data[0].shape[1]
		self.num_data = data[0].shape[0]
		self.config = config
		self.cov_W = cov_W
		self.cov_W_determinant = det(cov_W)
		self.cov_w_inv = inv(cov_W)
		self.mu_W = mu_W
		
		#self.sparsity_hyperparameter = alpha
		# self.testX = np.load('ls_nls_x_test.npy')
		# self.testY = np.load('ls_nls_y_test.npy')


	def multilaplace(self,x):

		temp =  np.dot(np.dot(x.T,self.cov_cat_inv),x)
		n = np.shape(x)[0]
		v = float(2-n)/2.0
		a = 2.0/((2*np.pi)**(n/2.0) * np.sqrt(self.cov_cat_determinant))
		return a * (temp/2)**(v/2.0) * sp.kv(v,np.sqrt(temp))


	def prior(self,w1,w2,sigma):
		w_prior = 0.0
		# prior of w is a gaussian with mean mu_W and covariance matrix cov_W
		#w_prior = sp.multivariate_normal.logpdf(w, mean=self.mu_W, cov=self.cov_W)
		
		#If sparsity is added we add the laplace prior
		if self.config['lambdapriors'] == 0:
			w_prior = w_prior + self.config['sparsity_hyperparam'] *np.linalg.norm(w1,ord=1)
		else:
			w_prior = w_prior + sp.multivariate_normal.logpdf(w1, mean=self.mu_W, cov=self.cov_W)	


		if self.config['catpriors'] == 0:
			w_prior = w_prior + self.config['sparsity_hyperparam'] *np.linalg.norm(w2,ord=1)
		else:
			w_prior = w_prior + sp.multivariate_normal.logpdf(w2, mean=self.mu_W, cov=self.cov_W)	



		return w_prior


	def likelihood(self,epsilon,w1,w2,sigma):

		lamb_feat, cat_feat = self.data

		p = vec_sigmoid(np.dot(lamb_feat,w1)+np.dot(cat_feat,w2)+ epsilon)		
		ep_like  =  sp.norm.logpdf(epsilon,scale=sigma)
		label_like = np.sum(vec_Bern_pdf(p,self.label))
		# if np.isnan(ep_like+label_like):
		# 	print 'likelihood:',ep_like
		return ep_like+label_like


	def posterior(self,epsilon,w1,w2,sigma):
		# print self.prior(w,sigma) + self.likelihood(epsilon,w,sigma), 'From posterior'
		return self.prior(w1,w2,sigma) + self.likelihood(epsilon,w1,w2,sigma)


	def assessment_on_test_data(self, w,sig):

		p = sigmoid(np.dot(self.testX, w) + sig)
		y_hat = [np.random.binomial(1, i) for i in p]

		cm = confusion_matrix(self.testY, y_hat)
		# print cr
		# print cm
		num_class0_miss = cm[0][1] 
		num_class1_miss = cm[1][0]
		total_miss_class = num_class0_miss + num_class1_miss
		ratio = total_miss_class/float(len(self.testY))
		return ratio

	def metropolis_hastings(self, iter=2000,sample_size=1000,scale=0.01):

		# w_init = np.random.multivariate_normal(self.mu_W, self.cov_W)
		#### FOR LS LIWC took abs of the matrix
		w_init = np.random.multivariate_normal(self.mu_W, self.cov_W)
		sigma_init = np.random.random()
		epsilon_init = np.random.normal(0,sigma_init)


		epsilon, w1,w2, sigma = epsilon_init, w_init,w_init, sigma_init
		samples_w1 = np.zeros((sample_size, self.num_cat))
		samples_w2 = np.zeros((sample_size, self.num_cat))
		samples_sigma = np.zeros((sample_size, 1))
		mis_class_ratios = []
		
		for i in range(iter+sample_size):
			mis_class_ratios.append(-1*self.likelihood(epsilon,w1,w2,sigma)/float(self.num_data))
			#Proposed next states
			epsilon_star = epsilon + np.random.normal(scale=scale)
			w1_star = w1 + np.random.normal(scale=scale, size=self.num_cat)
			w2_star = w2 + np.random.normal(scale=scale, size=self.num_cat)
			sigma_star = sigma
			prop_sigma_star = sigma + np.random.normal(scale=scale)
			if prop_sigma_star >= 0:
				sigma_star = prop_sigma_star

			#Accept/Reject with metropolis filter
			# print 'proposed', np.exp(self.posterior(epsilon_star,w_star,sigma_star))
			# print 'previous', np.exp(self.posterior(epsilon,w,sigma))
			# print 'Diff:',np.log(np.random.rand())

			if np.log(np.random.rand()) < (self.posterior(epsilon_star,w1_star,w2_star,sigma_star) - self.posterior(epsilon,w1,w2,sigma)):
				# print 'Accepting'
				epsilon,w1,w2,sigma = epsilon_star,w1_star,w2_star,sigma_star
			if i>=iter:
				samples_w1[i-iter] = w1
				samples_w2[i-iter] = w2
				samples_sigma[i-iter] = sigma




		# plt.plot(range(len(mis_class_ratios)),mis_class_ratios)
		# plt.xlabel('# Iteration')
		# plt.ylabel('Normalized Negative Likelyhood Loss')
		# plt.grid()
		#plt.show()
		return samples_w1,samples_w2, samples_sigma


	