from io import open
import pickle
import sys
sys.path.insert(0,'./zipped_files_from_www')
sys.path.insert(0,'./Boyu')
from posterior import *
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,confusion_matrix, roc_curve, auc, classification_report, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def random_split(low_list, notlow_list, split_ratio = 2.0/3, multiplier=1):

    num_low, num_notlow = len(low_list), len(notlow_list)
    low_train_indices, notlow_train_indices = np.random.choice(num_low,size = int(num_low*split_ratio),replace=False), np.random.choice(num_notlow,size = int(num_notlow*split_ratio),replace=False)
    train_low, test_low, train_notlow, test_notlow = [],[],[],[]
    for i in range(num_low):
        if i in low_train_indices:
            train_low.append(multiplier*low_list[i][1])
        else:
            test_low.append(multiplier*low_list[i][1])
            
    for i in range(num_notlow):
        if i in notlow_train_indices:
            train_notlow.append(multiplier*notlow_list[i][1])
        else:
            test_notlow.append(multiplier*notlow_list[i][1])
            
    train_lambda, train_label = np.array(train_low+train_notlow), np.array([1 for i in train_low]+[0 for j in train_notlow])
    test_lambda, test_label = np.array(test_low+test_notlow), np.array([1 for i in test_low]+[0 for j in test_notlow])

            
    return train_lambda, train_label, test_lambda, test_label


def plot_features(ls_list,nls_list,multiplier = 10**5):
    
    
    low_mean =multiplier* np.mean(np.array([elem[1] for elem in ls_list]),axis=0)
    notlow_mean =multiplier* np.mean(np.array([elem[1] for elem in nls_list]),axis=0)
    # print(low_mean)
    # print(notlow_mean)
    labels = ["c"+str(i+1) for i in range(27)]
    xaxis = np.arange(27)
    width = 0.3 

    fig, ax = plt.subplots()
    rects1 = ax.bar(xaxis - width/2, low_mean, width, label='Low')
    rects2 = ax.bar(xaxis + width/2, notlow_mean, width, label='Not Low')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('ISI params')
    ax.set_title('ISI for two groups')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(labels)
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


 # class ClassName(object):
 #               """docstring for ClassName"""
 #               def __init__(self, arg):
 #                   super(ClassName, self).__init__()
 #                   self.arg = arg
                             
        
    