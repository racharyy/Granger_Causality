from io import open
import pickle
import sys
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')
from posterior import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from helper import *
#from random_slice import *
cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]


ls_lambda, nls_lambda = load_pickle("../Boyu/lambda_vectors_with_user_ID.pkl")
psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
npsi_compound = npsi_compound[:37]+npsi_compound[38:]
# ls_lambda = [(elem[0],elem[1][27:]) for elem in ls_compound]
# nls_lambda = [(elem[0],elem[1][27:]) for elem in nls_compound]

#print(npsi_compound[37])
# for a,i in enumerate(npsi_compound):
# 	print(a,i[0])
# 	print(len(i[1]))
indx_lst = [0,2,3,4,6,7,9,13,25]
psi_lambda = [(elem[0],elem[1][27:]) for elem in psi_compound]
npsi_lambda = [(elem[0],elem[1][27:]) for elem in npsi_compound]
plot_features(ls_lambda,nls_lambda,indx_lst,multiplier=10**5)
#print(psi_lambda)

# ls_cv,nls_cv = load_pickle("../Boyu/cv_list.pkl")
# labels = [cats[i] for i in range(27)]
# xaxis = np.arange(27)
# width = 0.3 

# fig, ax = plt.subplots()
# rects1 = ax.bar(xaxis - width/2, ls_cv, width, label='Low self-esteem')
# rects2 = ax.bar(xaxis + width/2, nls_cv, width, label='Not low\nself-esteem')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('CV params')
# #ax.set_title('CV for two groups')
# ax.set_xticks(xaxis)
# ax.set_xticklabels(labels,rotation=90)
# ax.legend()

# fig.tight_layout() 
# plt.show()


#plot_features(ls_lambda,nls_lambda,op='log')