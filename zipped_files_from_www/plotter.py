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


ls_compound, nls_compound = load_pickle("../Boyu/compound_vectors_self_esteem.pkl")
psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
npsi_compound = npsi_compound[:37]+npsi_compound[38:]
ls_lambda = [(elem[0],elem[1][27:]) for elem in ls_compound]
nls_lambda = [(elem[0],elem[1][27:]) for elem in nls_compound]

print(npsi_compound[37])
# for a,i in enumerate(npsi_compound):
# 	print(a,i[0])
# 	print(len(i[1]))

psi_lambda = [(elem[0],elem[1][27:]) for elem in psi_compound]
npsi_lambda = [(elem[0],elem[1][27:]) for elem in npsi_compound]
plot_features(psi_lambda,npsi_lambda)