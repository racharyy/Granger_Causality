import numpy as np

from posterior import *



all_train = ['Authentic', 'WPS', 'Sixltr', 'they', 'auxverb', 'negate', 'adj', 'number', 'affect', 'posemo', 'anx', 'sad', 'family', 'friend', 'cogproc', 'insight', 'cause', 'differ', 'see', 'bio', 'body', 'health', 'drives', 'affiliation', 'achieve', 'risk', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'money', 'informal', 'swear', 'netspeak', 'nonflu', 'AllPunc', 'QMark', 'Dash', 'Apostro']
all_features = ["Filename","Segment","WC","Analytic","Clout","Authentic","Tone","WPS","Sixltr","Dic","function","pronoun","ppron","i","we","you","shehe","they","ipron","article","prep","auxverb","adverb","conj","negate","verb","adj","compare","interrog","number","quant","affect","posemo","negemo","anx","anger","sad","social","family","friend","female","male","cogproc","insight","cause","discrep","tentat","certain","differ","percept","see","hear","feel","bio","body","health","sexual","ingest","drives","affiliation","achieve","power","reward","risk","focuspast","focuspresent","focusfuture","relativ","motion","space","time","work","leisure","home","money","relig","death","informal","swear","netspeak","assent","nonflu","filler","AllPunc","Period","Comma","Colon","SemiC","QMark","Exclam","Dash","Quote","Apostro","Parenth","OtherP"]
all_train_index = []

for index,feature in enumerate(all_features):
	if feature in all_train:
		all_train_index.append(index)



ls_feature_data = np.loadtxt(
	"/home/anis/Dropbox/Data/takeout_data/train/low_self_esteem/LIWC2015_train_low_self_esteem.csv",
	skiprows=1,
	usecols=all_train_index,
	dtype='float32', 
	delimiter=','
)

print '======================='
print 'Train: Low self-esteem Data shape:', ls_feature_data.shape


class_label = [1]*len(ls_feature_data)
# # 1 means considered committing suicide
# # 0 means never considered committing suicide

# get non suicide subjects
non_ls_feature_data = np.loadtxt(
	"/home/anis/Dropbox/Data/takeout_data/train/not_low_self_esteem/LIWC2015_train_not_low_self_esteem.csv",
	skiprows=1,
	usecols=all_train_index,
	dtype='float32', 
	delimiter=','
)
# indices = np.random.randint(0,non_ls_feature_data.shape[0],20)
# non_ls_feature_data = non_ls_feature_data[indices]

print 'Train: Not low self-esteem Data shape:', non_ls_feature_data.shape

class_label = class_label + [0] * len(non_ls_feature_data)

# print len(class_label)
# print class_label
X = np.concatenate((ls_feature_data,non_ls_feature_data), axis=0)
y = np.array(class_label)

print 'X', X.shape
print 'y', y.shape

####################################
############ Test data #############
ls_feature_test_data = np.loadtxt(
	"/home/anis/Dropbox/Data/takeout_data/validation/low_self_esteem/LIWC2015_validation_low_self_esteem.csv",
	skiprows=1,
	usecols=all_train_index,
	dtype='float32', 
	delimiter=','
)
print 'Test: Suicide Data shape:', ls_feature_test_data.shape

test_class_label = [1]*len(ls_feature_test_data)

non_ls_feature_test_data = np.loadtxt(
	"/home/anis/Dropbox/Data/takeout_data/validation/not_low_self_esteem/LIWC2015_validation_not_low_self_esteem.csv",
	skiprows=1,
	usecols=all_train_index,
	dtype='float32', 
	delimiter=','
)
print 'Test: Non Suicide Data shape:',non_ls_feature_test_data.shape
print '======================='
test_class_label = test_class_label + [0] * len(non_ls_feature_test_data)

test_X = np.concatenate((ls_feature_test_data,non_ls_feature_test_data), axis=0)
test_y = np.array(test_class_label)

print 'test X:', test_X.shape
print 'test y:', test_y.shape
########################################################################

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

simMatrix2 = np.loadtxt(
    "ls_liwc_cov.csv",
    dtype='float32', 
    delimiter=','
)
print is_pos_def(simMatrix2)



cov_W = simMatrix2
cov_W_inv = inv(cov_W)

# cov_W = (1- simMatrix) + np.identity(len(cats))
mu_W = np.zeros(len(all_train))
print mu_W.shape
print cov_W.shape
print 'Shape: similarity matrix',simMatrix2.shape
print 'Shape: covariance matrix',cov_W.shape

bmodel = Bayes_model(X,y ,cov_W,mu_W)
samples_w, samples_sigma = bmodel.metropolis_hastings()

# np.savetxt('experiments/lsliwc/final/0.1/false/sample_w_0.1_false.csv', samples_w, delimiter=',')
# np.savetxt('experiments/lsliwc/final/0.1/false/sample_sigma_0.1_false.csv', samples_sigma, delimiter=',')
