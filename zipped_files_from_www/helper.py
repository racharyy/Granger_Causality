from io import open
import pickle
import sys
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')
from posterior import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


cats = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

def random_split(low_list, notlow_list, split_ratio = 2.0/3, multiplier=1):

    num_low, num_notlow = len(low_list), len(notlow_list)
    low_train_indices, notlow_train_indices = np.random.choice(num_low,size = int(num_low*split_ratio),replace=False), np.random.choice(num_notlow,size = int(num_notlow*split_ratio),replace=False)
    train_low, test_low, train_notlow, test_notlow = [],[],[],[]
    train_low_user, test_low_user, train_notlow_user, test_notlow_user = [],[],[],[]
    for i in range(num_low):
        if i in low_train_indices:
            #print(low_list[i])
            train_low.append(multiplier*low_list[i][1])
            train_low_user.append(multiplier*low_list[i][0])
        else:
            test_low.append(multiplier*low_list[i][1])
            test_low_user.append(multiplier*low_list[i][0])
            
    for i in range(num_notlow):
        if i in notlow_train_indices:
            train_notlow.append(multiplier*notlow_list[i][1])
            train_notlow_user.append(multiplier*notlow_list[i][0])
        else:
            test_notlow.append(multiplier*notlow_list[i][1])
            test_notlow_user.append(multiplier*notlow_list[i][0])
            
    train_lambda, train_label = np.array(train_low+train_notlow), np.array([1 for i in train_low]+[0 for j in train_notlow])
    test_lambda, test_label = np.array(test_low+test_notlow), np.array([1 for i in test_low]+[0 for j in test_notlow])

    train_user =  np.array(train_low_user+train_notlow_user)
    test_user =  np.array(test_low_user+test_notlow_user) 

    return train_lambda, train_label, test_lambda, test_label, train_user, test_user


def extract_index(ls,nls,train_user,test_user):

    train_lambda, train_label, test_lambda, test_label =  [],[],[],[]
    for data in ls:

        user, vec = data 
        if user in train_user:
            train_lambda.append(vec)
            train_label.append(1)
        elif user in test_user:
            test_lambda.append(vec)
            test_label.append(1)

    for data in nls:

        user, vec = data
        if user in train_user:
            train_lambda.append(vec)
            train_label.append(0)
        elif user in test_user:
            test_lambda.append(vec)
            test_label.append(0)


    return train_lambda, train_label, test_lambda, test_label


def plot_features(ls_list,nls_list,multiplier = 1,op='None'):
    
    if op=='exp':
        low_mean = np.mean(np.exp(multiplier*np.array([elem[1] for elem in ls_list])),axis=0)
        notlow_mean =np.mean(np.exp(multiplier*np.array([elem[1] for elem in nls_list])),axis=0)
    elif op == 'inv':
        low, notlow = [],[]
        for elem in ls_list:
            x= elem[1]
            for i in range(len(x)):
                if x[i] == 0:
                    x[i] = 1000
                else:
                    x[i] = 1.0/x[i]
            low.append(np.array(x))

        for elem in nls_list:
            x= elem[1]
            for i in range(len(x)):
                if x[i] == 0:
                    x[i] = 1000
                else:
                    x[i] = 1.0/x[i]
            notlow.append(np.array(x))
        low_mean,notlow_mean = np.mean(low,axis=0), np.mean(notlow,axis=0)

    elif op == 'log':

        low, notlow = [],[]
        for elem in ls_list:
            x= elem[1]
            for i in range(len(x)):
                if x[i] == 0:
                    x[i] = -8
                else:
                    x[i] = math.log(x[i])/math.log(10)
            low.append(np.array(x))

        for elem in nls_list:
            x= elem[1]
            for i in range(len(x)):
                if x[i] == 0:
                    x[i] = -8
                else:
                    x[i] = math.log(x[i])/math.log(10)
            notlow.append(np.array(x))
        low_mean,notlow_mean = np.mean(low,axis=0), np.mean(notlow,axis=0)
    
    else:
        low_mean =multiplier* np.mean(np.array([elem[1] for elem in ls_list]),axis=0)
        notlow_mean =multiplier* np.mean(np.array([elem[1] for elem in nls_list]),axis=0)
    # print(low_mean)
    # print(notlow_mean)
    labels = [cats[i] for i in range(27)]
    xaxis = np.arange(27)
    width = 0.3 

    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis - width/2, low_mean, width, label='Low')
    rects2 = ax.bar(xaxis + width/2, notlow_mean, width, label='Not Low')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('ISI params')
    ax.set_title('ISI for two groups')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(labels,rotation=90)
    ax.legend()
    
    fig.tight_layout() 
    plt.show()
 
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data



def sentiment_scores(sentence): 
    '''
    Given a sentence, this function returns a dictionary
    with the positive, negative, neutral and compound 
    sentiment. For example it returns a dic like this:

    {
        'pos': 0.487, 
        'compound': 0.431, 
        'neu': 0.513, 
        'neg': 0.0
    }
    '''
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 

    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict


def get_data_for_leave_one_out(X, y):
    '''
    X is the feature matrix
    y is the corresponding label
    '''
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    print(loo.get_n_splits(X))
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, y_train, X_test, y_test


# def plot_roc_auc_for_each_fold()

# if __name__ == '__main__':
#     X = np.random.normal(size=(100,3))
#     y = np.random.randint(2,size=100)
#     # print(X)
#     # print(y)
#     i=0
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     lr = LogisticRegression(random_state=0, solver='liblinear')
#     y_true, prob = [], []
#     for X_train, y_train, X_test, y_test in get_data_for_leave_one_out(X, y):
#         #print('Do trianing on the x train, y train and test on the x test and y_test')
#         # print(X_train.shape, y_train.shape)
#         # print(X_train,y_train)
#         # ******************************************
#         # prediction probabilities on test set
#         probas_ = lr.fit(X_train, y_train).predict_proba(X_test)
#         # ******************************************
#         #print(probas_)
#         y_true.append(y_test[0])
#         prob.append(probas_[:, 1][0])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(y_true, prob)
#     #print(y_test,  probas_[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)

#     aucs.append(roc_auc)
#     plt.plot(
#         fpr, 
#         tpr, 
#         lw=1, 
#         alpha=0.3
#         #label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
#     )

#     # i = i+ 1
#     plt.plot(
#         [0, 1],
#         [0, 1],
#         linestyle='--', 
#         lw=2, 
#         color='r',
#         label='Chance', 
#         alpha=.8
#     )

#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(
#         mean_fpr, 
#         mean_tpr, 
#         color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8
#     )

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(
#         mean_fpr,
#         tprs_lower,
#         tprs_upper,
#         color='grey',
#         alpha=.2,
#         label=r'$\pm$ 1 std. dev.')

#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()




# ls_compound, nls_compound = load_pickle("../Boyu/compound_vectors_self_esteem.pkl")
# psi_compound, npsi_compound = load_pickle('../Boyu/compound_vectors_psi.pkl')
# ls_lambda = [(elem[0],elem[1][27:]) for elem in ls_compound]
# nls_lambda = [(elem[0],elem[1][27:]) for elem in nls_compound]
# #print(len(ls_compound[0][1]))

# ls_joint = [(elem[0],elem[1][:27]*elem[1][27:]) for elem in ls_compound]
# nls_joint = [(elem[0],elem[1][:27]*elem[1][27:]) for elem in nls_compound]
# #plot_features(ls_lambda,nls_lambda)
# plot_features(ls_joint,nls_joint)


 # class ClassName(object):
 #               """docstring for ClassName"""
 #               def __init__(self, arg):
 #                   super(ClassName, self).__init__()
 #                   self.arg = arg
                             
 # def plot_():
 #     pass



    