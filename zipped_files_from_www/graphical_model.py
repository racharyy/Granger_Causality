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

# cov_search = np.loadtxt(
#     "similarity_matrix.csv",
#     dtype='float32', 
#     delimiter=','
# )

# mu_search = np.zeros(27) 

cov_search, mu_search = np.eye(10),np.zeros(10)


def model_fit_using_se(data,u_dim,method='mcmc', num_iter = 10,num_sample=1000):



	search_dim = data['search'].shape[1]
	num_obs = data['search'].shape[0]

	cov_u, mu_u = np.eye(u_dim),np.zeros(u_dim)
	cov_nlp, mu_nlp = np.eye(search_dim),np.zeros(search_dim)

	cov_nlp = np.loadtxt("similarity_matrix.csv",dtype='float32', delimiter=',')

	with pm.Model() as model:


		u = pm.MvNormal('u',mu=mu_u,cov=cov_u,shape=(num_obs,u_dim))
		search = data['search']#pm.MvNormal('search',mu=mu_search,cov=cov_search,observed=data['search'])

		#Incoming edge to self esteem
		u_se = pm.MvNormal('u_se',mu=mu_u,cov=cov_u,shape=u_dim)
		search_se = pm.MvNormal('search_se',mu=mu_nlp,cov=cov_nlp,shape=search_dim)

		#self esteem as a function of its parents
		se_mean = tt.nnet.nnet.sigmoid(tt.dot(search,search_se)+tt.dot(u,u_se))
		se = pm.Bernoulli('se',p=se_mean,observed=data['se'])

		#Incoming edge to suicide ideation
		u_si = pm.MvNormal('u_si',mu=mu_u,cov=cov_u,shape=u_dim)
		search_si = pm.MvNormal('search_si',mu=mu_nlp,cov=cov_nlp,shape=search_dim)
		#se_si_sig = pm.Normal('se_si_sig',mu=0,tau=1)
		se_si = pm.HalfNormal('se_si',sigma=1)#se_si_sig**2)

		si_mean = tt.nnet.nnet.sigmoid(tt.dot(search,search_si)+tt.dot(u,u_si)+se_si*se)
		si = pm.Bernoulli('si',p=si_mean,observed=data['si'])

		mf=pm.fit(n=num_iter)
		#trace = pm.sample()
		trace = mf.sample(num_sample)
		#trace = pm.sample()
		# se_mean = pm.Uniform('se_mean',lower=0,upper=1,size=num_obs)
		# se = pm.Bernoulli('se',p=se_mean, observed = data["se"])

		# si_me
		# si = pm.Bernoulli('si',p= ,observed = data["si"])

	return trace


# def model_fit_generative(data,bn_dim,method='mcmc', num_iter = 10,num_sample=1000):

# 	search_dim = data['search'].shape[1]
# 	num_obs = data['search'].shape[0]

# 	cov_bn, mu_bn = np.eye(u_dim),np.zeros(u_dim)
# 	cov_nlp, mu_nlp = np.eye(search_dim),np.zeros(search_dim)

# 	cov_nlp = np.loadtxt("similarity_matrix.csv",dtype='float32', delimiter=',')

# 	with pm.Model() as model:


# 		bn = pm.MvNormal('bn',mu=mu_bn,cov=cov_bn,shape=(num_obs,bn_dim))




def model_fit_basic(data,u_dim,method='mcmc', num_iter = 10,num_sample=1000,use_u=False):



	search_dim = data['search'].shape[1]
	num_obs = data['search'].shape[0]

	cov_u, mu_u = np.eye(u_dim),np.zeros(u_dim)
	cov_nlp, mu_nlp = np.eye(search_dim),np.zeros(search_dim)

	with pm.Model() as model:


		
		search = data['search']#pm.MvNormal('search',mu=mu_search,cov=cov_search,observed=data['search'])

		search_si = pm.MvNormal('search_si',mu=mu_nlp,cov=cov_nlp,shape=search_dim)
		si_mean = tt.dot(search,search_si) 

		if use_u:
			u = pm.MvNormal('u',mu=mu_u,cov=cov_u,shape=(num_obs,u_dim))
			u_si = pm.MvNormal('u_si',mu=mu_u,cov=cov_u,shape=u_dim)
			si_mean = si_mean +tt.dot(u,u_si)
		
			
		#Incoming edge to suicide ideation	
		si_mean = tt.nnet.nnet.sigmoid(si_mean)
		si = pm.Bernoulli('si',p=si_mean,observed=data['si'])

		mf=pm.fit(n=num_iter)
		#trace = pm.sample()
	trace = mf.sample(num_sample)	

	return trace


key_list = ['u','u_se','search_se','u_si','search_si','se_si']
# data = load_pickle('data_for_graphical_model.pkl')
# print(data.keys())


def predict_gm(test_data,param_dic,u_dim,u_sample_num=100):

	si_prob = 0

	num_test_data = test_data['search'].shape[0]
	for u in range(u_sample_num):

		u = np.random.normal(size=(num_test_data,u_dim))
		search = test_data['search']
		se = test_data['se']

		# u_prob = norm.pdf(u)
		# u_prob = np.prod(u_prob,axis=1)

		u_si = param_dic['u_si']
		search_si = param_dic['search_si']
		se_si = param_dic['se_si']

		si_mean = expit(np.dot(search,search_si)+np.dot(u,u_si)+se_si*se)

		si_prob = si_prob + si_mean

	return si_prob/float(u_sample_num)

#train_data, test_data = load_pickle('../data_split.pkl')





def train_and_test(train_data_dict,test_data_dict,u_dim,method='not_basic'):

	
	param_dic = {}
	if method =='basic':
		trace = model_fit_basic(train_data_dict,u_dim=u_dim,num_iter=20000)
	elif method=='generative':
		trace = model_fit_generative(train_data_dict,u_dim=u_dim,num_iter=10000,num_sample=200)
	else:
		trace = model_fit_using_se(train_data_dict,u_dim=u_dim,num_iter=10000,num_sample=1000)
	n=trace['search_si'].shape[0]
	f1s = []
	for i in range(n):
		# if i%100==0:
		# 	print(i)
		for key in key_list:
			param_dic[key] = trace[key][i]
		#print(trace[key].shape)

	#print(param_dic)
		prob=predict_gm(test_data_dict,param_dic,u_dim,u_sample_num=1000)
		prediction = np.random.binomial(1,prob)
		cr = classification_report(test_data_dict['si'], prediction)

		avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()[-4:]
		f1s.append(float(avg_f1))
	#print(f1s)
	median_f1 = np.median(np.array(f1s))
	mean_f1 = np.mean(np.array(f1s))
	return trace,median_f1,mean_f1
	#print(cr)




def run():
	data = load_pickle('data_for_graphical_model.pkl')
	N= len(data['se'])

	good_set = [] 
	good_num = 0
	median_f1_ar, mean_f1_ar = [], []
	for i in range(100):


		train_index = np.random.choice(N,int(0.8*N),replace=False)
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



		trace,median_f1,mean_f1 = train_and_test(train_data_dict,test_data_dict,12)
		median_f1_ar.append(median_f1)
		mean_f1_ar.append(mean_f1)
		print(median_f1,mean_f1)
		if median_f1 >= 0.67:
			good_num = good_num+1
			print(good_num)
			dic ={}
			dic['trace'] = trace
			dic['train_index'] = train_index
			good_set.append(dic)
	pickle.dump(good_set,open('good_set_gen.pkl','wb'))
	print(np.mean(np.array(median_f1_ar)),' : mean of the medians')
	print(np.mean(np.array(mean_f1_ar)),' : mean of the means')
	#print(median_f1,mean_f1)
# num_obs=40
# search_dim=10
# data = {}
# data['search'] = np.random.normal(size=(num_obs,search_dim))
# p= 0.8*np.ones(num_obs)
# data['se'] = np.random.binomial(1,p)
# data['si'] = np.random.binomial(1,p)
# #print(data['se'])
# trace = model_fit_basic(data,5)



