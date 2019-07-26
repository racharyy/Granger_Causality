import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp 



old_test_ls_nls_X = np.load('ls_nls_x_test.npy')
old_test_ls_nls_y = np.load('ls_nls_y_test.npy')

old_ls_nls_X = np.load('ls_nls_x.npy')
old_ls_nls_y = np.load('ls_nls_y.npy')

old_ls_nls_X = np.concatenate( ( old_ls_nls_X[:49], old_ls_nls_X[50:]  ), axis=0)
old_ls_nls_y = np.concatenate( ( old_ls_nls_y[:49], old_ls_nls_y[50:]  ), axis=0)

test_ls = np.sum(old_test_ls_nls_y)
train_ls = np.sum(old_ls_nls_y)

total_ls_X = np.concatenate((old_ls_nls_X[:train_ls],old_test_ls_nls_X[:test_ls]),axis=0)
total_nls_X = np.concatenate((old_ls_nls_X[train_ls:],old_test_ls_nls_X[test_ls:]),axis=0)



total_ls_y = np.concatenate((old_ls_nls_y[:train_ls],old_test_ls_nls_y[:test_ls]),axis=0)
total_nls_y = np.concatenate((old_ls_nls_y[train_ls:],old_test_ls_nls_y[test_ls:]),axis=0)

total_ls_len = (total_ls_X).shape[0]
total_nls_len = (total_nls_X).shape[0]

def our_slice(training_ls,training_nls):

	train_ls_ind=np.random.choice(total_ls_len,size= training_ls,replace=False)
	train_nls_ind=np.random.choice(total_nls_len,size= training_nls,replace=False)


	ls_nls_y = []
	ls_nls_X = []

	test_ls_nls_y = []
	test_ls_nls_X = []


	for idx in xrange(total_ls_len):
		
		if idx in train_ls_ind:
			ls_nls_X.append(total_ls_X[idx])
			ls_nls_y.append(total_ls_y[idx])
	
		else:

			test_ls_nls_X.append(total_ls_X[idx])
			test_ls_nls_y.append(total_ls_y[idx])


	for idx in xrange(total_nls_len):
		
		if idx in train_nls_ind:
			ls_nls_X.append(total_nls_X[idx])
			ls_nls_y.append(total_nls_y[idx])
	
		else:

			test_ls_nls_X.append(total_nls_X[idx])
			test_ls_nls_y.append(total_nls_y[idx])

	return np.array(ls_nls_X),np.array(ls_nls_y),np.array(test_ls_nls_X),np.array(test_ls_nls_y)



	
