import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as sp
import math



def Bern_pdf(p,l):
	assert (l==0 or l==1)
	if p == 0:
		p = 0.0001
	if p == 1:
		p = 0.9999
	return l*math.log(p)+(1-l)*math.log(1-p)
vec_Bern_pdf = np.vectorize(Bern_pdf)

def sigmoid(x):
	# print 'at sigmoid', 1.0/(1+np.exp(-x))
	return 1.0/(1+np.exp(-x))
vec_sigmoid = np.vectorize(sigmoid)