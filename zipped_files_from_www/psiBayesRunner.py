import numpy as np

from posterior import *
cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

test_psi_npsi_X = np.load('psi_npsi_x_test.npy')
test_psi_npsi_y = np.load('psi_npsi_y_test.npy')

psi_npsi_X = np.load('psi_npsi_x.npy')
psi_npsi_y = np.load('psi_npsi_y.npy')

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


simMatrix = np.load('sim_mat.npy')

print simMatrix.shape
print is_pos_def(simMatrix)
print '=================='
simMatrix2 = np.loadtxt(
    "similarity_matrix.csv",
    dtype='float32', 
    delimiter=','
)
print is_pos_def(simMatrix2)

print '++++++++++++++++++'
print psi_npsi_X.shape
print psi_npsi_y.shape
print '++++++++++++++++++'
cov_W = simMatrix2
cov_W_inv = inv(cov_W)

# cov_W = (1- simMatrix) + np.identity(len(cats))
mu_W = np.zeros(len(cats))
print mu_W.shape
print cov_W.shape
print 'Shape: similarity matrix',simMatrix.shape
print 'Shape: covariance matrix',cov_W.shape

# bmodel = Bayes_model(psi_npsi_X,psi_npsi_y ,cov_W,mu_W)
# samples_w, samples_sigma = bmodel.metropolis_hastings()
# # samples_w = [cats] + list(samples_w)


# np.savetxt('experiments/psi/sample_w_0.0001_false.csv', samples_w, delimiter=',')
# np.savetxt('experiments/psi/sample_sigma_0.0001_false.csv', samples_sigma, delimiter=',')

