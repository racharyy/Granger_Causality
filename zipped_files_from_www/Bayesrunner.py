import numpy as np
from random_slice import *
from posterior import *
cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]
test_ls_nls_X = np.load('ls_nls_x_test.npy')
test_ls_nls_y = np.load('ls_nls_y_test.npy')

# ls_nls_X = np.load('ls_nls_x.npy')
# ls_nls_y = np.load('ls_nls_y.npy')



training_ls,training_nls = 31,24

ls_nls_X,ls_nls_y,test_ls_nls_X,test_ls_nls_y=our_slice(training_ls,training_nls)


np.save('ls_nls_X_v2.npy',ls_nls_X)
np.save('ls_nls_y_v2.npy',ls_nls_y)
np.save('ls_nls_X_test_v2.npy',test_ls_nls_X)
np.save('ls_nls_y_test_v2.npy',test_ls_nls_y)




print(ls_nls_y)
print('===================')
print(test_ls_nls_y)




# old_test_ls_nls_X = np.load('ls_nls_x_test.npy')
# old_test_ls_nls_y = np.load('ls_nls_y_test.npy')

# old_ls_nls_X = np.load('ls_nls_x.npy')
# old_ls_nls_y = np.load('ls_nls_y.npy')

# old_ls_nls_X = np.concatenate( ( old_ls_nls_X[:49], old_ls_nls_X[50:]  ), axis=0)
# old_ls_nls_y = np.concatenate( ( old_ls_nls_y[:49], old_ls_nls_y[50:]  ), axis=0)


# # print old_test_ls_nls_y
# # print 'tr',old_ls_nls_y


# delta_nls = 4
# delta_ls = 6

# ls_nls_y = np.concatenate((old_test_ls_nls_y[:delta_ls], old_ls_nls_y[:-delta_nls]),axis=0)
# ls_nls_X = np.concatenate((old_test_ls_nls_X[:delta_ls],old_ls_nls_X[:-delta_nls]),axis=0)
# # Done Train split

# nls_delta_xs = old_ls_nls_X[-delta_nls:]
# nls_delta_ys = old_ls_nls_y[-delta_nls:]

# test_ls_nls_X = np.concatenate((old_test_ls_nls_X[delta_ls:],nls_delta_xs),axis=0)
# test_ls_nls_y = np.concatenate((old_test_ls_nls_y[delta_ls:],nls_delta_ys),axis=0)

# for index,row in enumerate(old_ls_nls_X):
# 	print index, np.sum(row), '<---'

print(np.sum(ls_nls_y))
print(np.sum(test_ls_nls_y))
print('===========')
print('train X',ls_nls_X.shape)
print('train y',ls_nls_y.shape)
print('test X',test_ls_nls_X.shape)
print('test y',test_ls_nls_y.shape)
print('===========')

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


simMatrix = np.load('sim_mat.npy')

print(simMatrix.shape)
print(is_pos_def(simMatrix))
print '=================='
simMatrix2 = np.loadtxt(
    "similarity_matrix.csv",
    dtype='float32', 
    delimiter=','
)
print(is_pos_def(simMatrix2))

cov_W = simMatrix2
cov_W_inv = inv(cov_W)

# cov_W = (1- simMatrix) + np.identity(len(cats))
mu_W = np.zeros(len(cats))
print mu_W.shape
print cov_W.shape
print 'Shape: similarity matrix',simMatrix.shape
print 'Shape: covariance matrix',cov_W.shape

bmodel = Bayes_model(ls_nls_X,ls_nls_y ,cov_W,mu_W)
samples_w, samples_sigma = bmodel.metropolis_hastings()


# samples_w = [cats] + list(samples_w)

# TRUE
# np.savetxt('experiments/sample_w_0.01_true.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.01_true.csv', samples_sigma, delimiter=',')

# np.savetxt('experiments/sample_w_0.1_true.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.1_true.csv', samples_sigma, delimiter=',')

# np.savetxt('experiments/sample_w_0.001_true.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.001_true.csv', samples_sigma, delimiter=',')

# False


# np.savetxt('experiments/sample_w_0.01_false.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.01_false.csv', samples_sigma, delimiter=',')

# np.savetxt('experiments/sample_w_0.1_false.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.1_false.csv', samples_sigma, delimiter=',')

# np.savetxt('experiments/sample_w_0.001_false.csv', samples_w, delimiter=',')
# np.savetxt('experiments/sample_sigma_0.001_false.csv', samples_sigma, delimiter=',')


np.savetxt('sample_w.csv', samples_w, delimiter=',')
np.savetxt('sample_sigma.csv', samples_sigma, delimiter=',')

