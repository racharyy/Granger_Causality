import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, entropy
from scipy.special import kl_div
import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import math
def  term_kl(p,q):
    if q==0 or p==0:
        return 0
    else:
        # print p,q,'========'
        return np.sum(p * math.log(p / q))
    
vec_term_kl = np.vectorize(term_kl)


def entropy_multi(p, q):
    return np.sum(vec_term_kl(p,q), axis=0)

def entropy_single(p):
    return np.sum(p * np.log(p), axis=0)

def KLD(pk, qk):
    # arraynise
    pk = np.asarray(pk)
    # normalise
    pk = 1.0*pk / np.sum(pk, axis=0)
    # check to decide if we apply single or multi entorpy
    if qk is None:
        return np.sum(entropy_single(pk), axis=0)
    else:
        # arraynise
        qk = np.asarray(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0*qk / np.sum(qk, axis=0)
        return np.sum(entropy_multi(pk, qk), axis=0)





ls_nls_X = np.load('ls_nls_x.npy')
ls_nls_y = np.load('ls_nls_y.npy')

print ls_nls_y
print ls_nls_y.shape

ls = []
nls = []

for index in range(len(ls_nls_y)):
    label = ls_nls_y[index]

    # if ls
    if label == 1:
        ls.append(ls_nls_X[index])
    else:
        nls.append(ls_nls_X[index])


ls = np.array(ls)
nls = np.array(nls)
print ls.shape
print nls.shape

stacked = np.concatenate((ls, nls), axis=0)
print stacked.shape

ls_nls_matrix = np.zeros(((len(ls)+len(nls)),(len(ls)+len(nls) )))


# #ls ls
for i,rowi in enumerate(stacked):
    for j,rowj in enumerate(stacked):
        # kl = entropy(row_ls, row_nls)
        # print 'kl entropy', kl
        if np.sum(rowi) == 1 and np.sum(rowj) == 1:
            kl_mine = np.linalg.norm(rowi-rowj)
            kl_mine = KLD(rowi,rowj)
            ls_nls_matrix[i][j] = kl_mine


# for ls_idx,row_ls in enumerate(ls):
#     for nls_index,row_nls in enumerate(nls):
#         # kl = entropy(row_ls, row_nls)
#         # print 'kl entropy', kl
#         if np.sum(row_nls) == 1 and np.sum(row_ls) == 1:
#             kl_mine = KLD(row_ls,row_nls)
#             ls_nls_matrix[ls_idx][nls_index] = kl_mine

print ls_nls_matrix
print ls_nls_matrix.shape
# plt.matshow(ls_nls_matrix)
sns.heatmap(ls_nls_matrix,cmap="YlGnBu")
# plt.title(ylabel+' Search Category Trajectory (by %)')
# plt.ylabel(ylabel)
# plt.xticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=90,fontsize=8)
# plt.yticks(fontsize=6)
# plt.subplots_adjust(bottom=0.3)
# plt.tight_layout()
# plt.colorbar()
plt.show()
# # data prep
# x = np.linspace(-10.0, 10.0, 1000)

# # graph setting
# plt.figure(figsize=(12,8))

# # gradually shift the distribution
# for i in np.arange(3):
#     for j in np.arange(3):
#         # index to shift
#         index = i*3 + j
#         # probabilistic distribution function
#         p = norm.pdf(x, loc=0, scale=1)
#         q = norm.pdf(x, loc=index*0.5, scale=1)
#         # mean of them
#         m = (p+q)/2
#         # scipy entropy
#         kl = entropy(p, q)
#         kl_mine = KLD(p,q)
#         # checker
#         print("my_KL: ", "{0:.2f}".format(kl_mine), "scipy_KL: ", "{0:.2f}".format(kl))
#         # prep for js divergence
#         kl_pm = KLD(p, m)
#         kl_qm = KLD(q, m)
#         js = (kl_pm + kl_qm)/2
#         plt.subplot(3,3,i*3+j+1)
#         plt.fill_between(x, m, facecolor="y", alpha=0.2)
#         plt.fill_between(x, p, facecolor="b", alpha=0.2)
#         plt.fill_between(x, q, facecolor="r", alpha=0.2)
#         plt.xlim(-5, 7)
#         plt.ylim(0,0.45)
#         plt.title("KLD:{:>.3f}".format(kl) + ",   JSD:{:>.3f}".format(js))
#         plt.tick_params(labelbottom="off")
#         plt.tick_params(labelleft="off")

# plt.subplots_adjust(wspace=0.1, hspace=0.5)
# plt.show()