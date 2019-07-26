import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib as plt
from numpy.linalg import inv

mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])
cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

def circle(x, y):
    return (x-1)**2 + (y-2)**2 - 3**2


def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(epsilon_init,w_init, sigma_init, iter=1000):
    epsilon, w, sigma = epsilon_init, w_init, sigma_init
    samples_w = np.zeros((iter, 2))

    for i in range(iter):
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
        samples[i] = np.array([x, y])

    return samples



if __name__ == '__main__':

    simMatrix = np.loadtxt(
        "similarity_matrix.csv",
        dtype='float32', 
        delimiter=','
    )
    covM = (1- simMatrix) + np.identity(len(cats))
    print 'Shape: similarity matrix',simMatrix.shape
    print 'Shape: covariance matrix',covM.shape


    # samples = metropolis_hastings(circle, iter=10000)
    # sns.jointplot(samples[:, 0], samples[:, 1])

    # samples = metropolis_hastings(pgauss, iter=10000)
    # sns.jointplot(samples[:, 0], samples[:, 1])


    # np.save('psi_npsi_x_test', test_psi_npsi_X)
    # np.save('psi_npsi_y_test.npy', test_psi_npsi_y)

    # np.save('psi_npsi_x.npy', psi_npsi_X)
    # np.save('psi_npsi_y.npy', psi_nsi_y)

    test_ls_nls_X = np.load('ls_nls_x_test.npy')
    test_ls_nls_y = np.load('ls_nls_y_test.npy')

    ls_nls_X = np.load('ls_nls_x.npy')
    ls_nls_y = np.load('ls_nls_y.npy')

    print ls_nls_X.shape
    print ls_nls_y.shape