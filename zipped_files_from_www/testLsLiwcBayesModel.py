import glob
import numpy as np
from auxilary import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
import random
import matplotlib.pyplot as plt
import pickle

random.seed(10)
cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]
all_train = ['Authentic', 'WPS', 'Sixltr', 'they', 'auxverb', 'negate', 'adj', 'number', 'affect', 'posemo', 'anx', 'sad', 'family', 'friend', 'cogproc', 'insight', 'cause', 'differ', 'see', 'bio', 'body', 'health', 'drives', 'affiliation', 'achieve', 'risk', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'money', 'informal', 'swear', 'netspeak', 'nonflu', 'AllPunc', 'QMark', 'Dash', 'Apostro']
all_features = ["Filename","Segment","WC","Analytic","Clout","Authentic","Tone","WPS","Sixltr","Dic","function","pronoun","ppron","i","we","you","shehe","they","ipron","article","prep","auxverb","adverb","conj","negate","verb","adj","compare","interrog","number","quant","affect","posemo","negemo","anx","anger","sad","social","family","friend","female","male","cogproc","insight","cause","discrep","tentat","certain","differ","percept","see","hear","feel","bio","body","health","sexual","ingest","drives","affiliation","achieve","power","reward","risk","focuspast","focuspresent","focusfuture","relativ","motion","space","time","work","leisure","home","money","relig","death","informal","swear","netspeak","assent","nonflu","filler","AllPunc","Period","Comma","Colon","SemiC","QMark","Exclam","Dash","Quote","Apostro","Parenth","OtherP"]
all_train_index = []

for index,feature in enumerate(all_features):
	if feature in all_train:
		all_train_index.append(index)


learned_parameters = [
	# ("experiments/sample_sigma_0.001_false.csv", "experiments/sample_w_0.001_false.csv"),
	# ("experiments/sample_sigma_0.1_false.csv", "experiments/sample_w_0.1_false.csv"),
	# ("experiments/sample_sigma_0.01_false.csv", "experiments/sample_w_0.01_false.csv"),
	# ("experiments/sample_sigma_0.001_true.csv", "experiments/sample_w_0.001_true.csv"),
	# ("experiments/sample_sigma_0.1_true.csv","experiments/sample_w_0.1_true.csv"),
	# ("experiments/sample_sigma_0.01_true.csv", "experiments/sample_w_0.01_true.csv"),
	# ('experiments/lsliwc/sample_sigma_0.01_false.csv','experiments/lsliwc/sample_w_0.01_false.csv')
	# ('experiments/lsliwc/sample_sigma_0.1_false.csv','experiments/lsliwc/sample_w_0.1_false.csv')
	# ('experiments/lsliwc/sample_sigma_0.01_true.csv','experiments/lsliwc/sample_w_0.01_true.csv')
	('experiments/lsliwc/final/0.1/false/sample_sigma_0.1_false.csv','experiments/lsliwc/final/0.1/false/sample_w_0.1_false.csv')

]


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
###############

test_ls_nls_X = test_X
test_ls_nls_y = test_y
print '====='
print test_ls_nls_y

for sig_file, w_file in learned_parameters:
	sampled_sigs = np.loadtxt(sig_file, delimiter=',')
	sampled_ws = np.loadtxt(w_file, delimiter=',')

	print '****'
	print sig_file
	print w_file
	print '****'
	# column means
	# sampled_ws is 500 x 27
	w = np.median(sampled_ws, axis=0)
	# sig = sampled_sigs[1]
	sig = np.median(sampled_sigs)
	print 'W',w.shape
	print 'test X', test_ls_nls_X.shape

	num_simulation = 200
	
	precisions = []
	recalls = []
	f1s = []

	total_miss = []
	class1_precisions = []
	class0_precisions = []
	
	class1_f1s = []
	class0_f1s = []

	class1_recalls = []
	class0_recalls = []
	
	class1_miss = []
	class0_miss = []

	result_iterations = {}

	for iter_num in range(num_simulation):
		p = sigmoid(np.dot(test_ls_nls_X, w) + sig)
		y_hat = [np.random.binomial(1, i) for i in p]

		for i in range(len(y_hat)):
			print 'predicted:', y_hat [i], '\t\t true:',test_ls_nls_y[i]
		print '============================='
		# print y_hat
		cr = classification_report(test_ls_nls_y, y_hat)
		cm = confusion_matrix(test_ls_nls_y, y_hat)
		print cr
		# print cm
		num_class0_miss = cm[0][1] 
		num_class1_miss = cm[1][0]

		avg_prec, avg_recal,avg_f1,_ = cr.split('avg / total')[-1].strip().split()
		
		clf_rep = precision_recall_fscore_support(test_ls_nls_y, y_hat)
		out_dict = {
			"precision" :clf_rep[0].round(2)
			,"recall" : clf_rep[1].round(2)
			,"f1-score" : clf_rep[2].round(2)
			,"support" : clf_rep[3]
		}
		# print out_dict
		class0_recalls.append(out_dict['recall'][0])
		class0_f1s.append(out_dict['f1-score'][0])
		class0_precisions.append(out_dict['precision'][0])
		class0_miss.append(num_class0_miss)

		class1_recalls.append(out_dict['recall'][1])
		class1_f1s.append(out_dict['f1-score'][1])
		class1_precisions.append(out_dict['precision'][1])
		class1_miss.append(num_class1_miss)
		total_miss.append(num_class0_miss+num_class1_miss)

		precisions.append(float(avg_prec))
		recalls.append(float(avg_recal))
		f1s.append(float(avg_f1))

		result_iterations[iter_num] = [(avg_prec, avg_recal,avg_f1), out_dict]
	
	exp_name = sig_file.split('/')[-1][:-4]
	plot_name = 'avg.png'
	plt.plot(range(len(precisions)), precisions, label='avg-prec')
	plt.plot(range(len(recalls)), recalls, label='avg-recall')
	plt.plot(range(len(f1s)), f1s,label='avg-f1')
	plt.legend()
	plt.title('Avg precision, recall, f1 scores')
	plt.xlabel('#iterations')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name+'_'+plot_name)
	plt.close()

	plot_name = 'miss.png'
	# misclassifications:
	plt.plot(range(len(total_miss)), total_miss,label='total miss')
	plt.plot(range(len(class0_miss)), class0_miss,label='class 0 miss')
	plt.plot(range(len(class1_miss)), class1_miss,label='class 1 miss')
	plt.legend()
	plt.xlabel('#iterations')
	plt.title('Misclassifications')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name+'_'+plot_name)
	plt.close()

	plot_name = 'class0.png'
	# class 0 level
	plt.plot(range(len(class0_precisions )),class0_precisions ,label = 'class 0 prec')
	plt.plot(range(len(class0_recalls )), class0_recalls,label = 'class 0 recal')
	plt.plot(range(len(class0_f1s)),class0_f1s ,label = 'class 0 f1')
	plt.xlabel('#iterations')
	plt.legend()
	plt.title('Class 0 metrics')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name+'_'+plot_name)
	plt.close()

	plot_name = 'class1.png'
	# class 1 level
	plt.plot(range(len(class1_precisions )),class1_precisions ,label = 'class 1 prec')
	plt.plot(range(len(class1_recalls )), class1_recalls,label = 'class 1 recal')
	plt.plot(range(len(class1_f1s)),class1_f1s ,label = 'class 1 f1')
	plt.xlabel('#iterations')
	plt.legend()
	plt.title('Class 1 metrics')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name+'_'+plot_name)
	plt.close()
	pickle.dump(result_iterations, open('tuning/liwc_ls/final/0.1/false/'+exp_name+'.pickle','wb'))


	# plotting histogram
	plot_name = '_total_miss_hist.png'
	plt.hist(total_miss,bins=23)
	plt.title('total missclassification')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()


	plot_name = '_class0_miss_hist.png'
	plt.hist(class0_miss,bins=23)
	plt.title('class 0 (not LS) miss')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()

	plot_name = '_class1_miss_hist.png'
	plt.hist(class1_miss,bins=23)
	plt.title('class 1 (LS) miss')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()


	plot_name = '_avgf1_hist.png'
	plt.hist(f1s,bins=100)
	plt.title('f1s')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()

	plot_name = '_class0f1_hist.png'
	plt.hist(class0_f1s,bins=100)
	plt.title('class 0 (not LS) f1s')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()

	plot_name = '_class1f1_hist.png'
	plt.hist(class1_f1s,bins=100)
	plt.title('class 1 (LS) f1s')
	plt.savefig('tuning/liwc_ls/final/0.1/false/'+exp_name +plot_name)
	plt.close()



	print '=================='
	# print result_iterations
	print '==========================================='
	
	