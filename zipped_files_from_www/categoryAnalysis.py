import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import seaborn as sns

cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

ls_users_cat_data = np.loadtxt(open('ls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,28)) )
print ls_users_cat_data.shape

nls_users_cat_data = np.loadtxt(open('nls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,30)) )
print nls_users_cat_data.shape

def normalize(metric, s_metric, ns_metric):
	s_metric = np.array(s_metric)
	ns_metric = np.array(ns_metric)
	# Normalize the values
	if np.sum(s_metric) == 0:
		s_metric = 0.0001 + s_metric
	else:
		s_metric = s_metric/np.linalg.norm(s_metric, ord=np.inf, axis=0, keepdims=True)
	
	if np.sum(ns_metric) == 0:
		ns_metric = 0.0001 + ns_metric
	else:
		ns_metric = ns_metric/np.linalg.norm(ns_metric, ord=np.inf, axis=0, keepdims=True)



	return metric, np.array(s_metric), np.array(ns_metric)

def calc_stats(X, Y, print_=False, fig_=None):
    """ returns: X.mean,
                 Y.mean,
                 t-test_p, 
                 Mann-Whitney test_p, 
                 Cohens_d """
    
    t,t_test_p = stats.ttest_ind(X, Y, axis=0, equal_var=False)
    ks,k_test_p = stats.ks_2samp(X, Y)

    try:
        mw,mw_p = stats.mannwhitneyu(X, Y)
    except ValueError:
        mw,mw_p = np.NAN,np.NAN

    n_tot = len(X) + len(Y)
    x_var = X.var(ddof=0)
    y_var = Y.var(ddof=0)
    pooled_var = (len(X)*x_var + len(Y)*y_var) / n_tot
    
    if(pooled_var ==0):
        cohens_d = np.nan
    else:
        cohens_d = (Y.mean() - X.mean()) / np.sqrt(pooled_var)
        

    my_stats = {
    	'x-mean':X.mean(),
    	'x-var':x_var,
    	'y-mean':Y.mean(),
    	'y-var': y_var,
    	't-test-pval':t_test_p/10.0, 
    	'mw-test-pval':mw_p, 
    	'cohens-d':cohens_d,
    	'kalmogorov':(ks,k_test_p)
    }
    return my_stats



def createHeatmap():
    bars = sorted(["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"])
    
    ls_users_cat_data = np.loadtxt(open('ls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,28)) )
    print ls_users_cat_data.shape

    nls_users_cat_data = np.loadtxt(open('nls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,30)) )
    print nls_users_cat_data.shape


    sns.heatmap(ls_users_cat_data, cmap="YlGnBu")
    # sns.heatmap(ls_users_cat_data, cmap="Greens")
    plt.yticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=0)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('user id of low self-esteem subjects')
    plt.ylabel('Categories')
    plt.title('Search categories distribution of \nparticipants with Low self-esteem')
    plt.tight_layout()
    plt.show()
    plt.close()

    sns.heatmap(nls_users_cat_data, cmap="YlGnBu")
    plt.yticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=0)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('user id of not low self-esteem subjects')
    plt.ylabel('Categories')
    plt.title('Search categories distribution of \nparticipants with not Low self-esteem')
    plt.tight_layout()
    plt.show()
    plt.close()

    '''
    psi_users_cat_data = np.loadtxt(open('si_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,14)) )
    print psi_users_cat_data.shape

    nsi_users_cat_data = np.loadtxt(open('nsi_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,42)) )
    print nsi_users_cat_data.shape


    sns.heatmap(psi_users_cat_data, cmap="YlGnBu")
    plt.yticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=0)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('user id of subjects \nwith prior suicide ideation')
    plt.ylabel('Categories')
    plt.title('Search categories distribution of \nparticipants with prior suicide ideation condition')
    plt.tight_layout()
    plt.show()
    plt.close()

    sns.heatmap(nsi_users_cat_data, cmap="YlGnBu")
    plt.yticks([r + 0.3 for r in range(len(cat_order))], cat_order, color='black',rotation=0)
    plt.subplots_adjust(left=0.2)
    plt.xlabel('user id of subjects \nwith no prior suicide ideation')
    plt.ylabel('Categories')
    plt.title('Search categories distribution of \nparticipants with no prior suicide ideation condition')
    plt.tight_layout()
    plt.show()
    plt.close()
    '''



    # ls_height = [ls_users_cat_data[i] for i,k in enumerate(bars)]
    # y_pos = np.arange(len(bars))

# print len(cat_order)
# for index, category in enumerate(cat_order):

    # category, normalized_ls, normalized_nls = normalize(category, ls_users_cat_data[index], nls_users_cat_data[index])
    # metrics = calc_stats(ls_users_cat_data[index],nls_users_cat_data[index])
    # print category,',',metrics['x-mean'], ',',metrics['y-mean'], ',',metrics['x-var'],',', metrics['y-var'], ',',metrics['kalmogorov']
    # print category, metrics['t-test-pval']


def create_boxplots(ipv_nipv_features_dic, ssd_cats, class1, class2):
    sns.set(style="ticks", palette="muted")
    columns = ['Search category', 'value', 'Subject type']
    rows = []
    for key,value in ipv_nipv_features_dic.iteritems():

        sy, nsy = value
        if key in ssd_cats:
            for i in sy:
                rows.append([key, float(i), class1 ])

            for j in nsy:
                rows.append([key, float(j), class2 ])
    rows = np.array(rows)

    ipv_nipv_dataframe = pd.DataFrame(data=rows, columns=columns)
    ipv_nipv_dataframe.to_csv(class1+'_'+class2+'.csv')
    ipv_nipv_dataframe = pd.read_csv(class1+'_'+class2+'.csv')
    ipv_nipv_dataframe.drop(['Unnamed: 0'], axis=1)
    

    ax = sns.boxplot(y="Search category", x="value", hue="Subject type",showfliers=False,data=ipv_nipv_dataframe, linewidth=1.5)
    # ax = sns.swarmplot(y="LIWC category", x="Attribute value", hue="Subject type",data=ipv_nipv_dataframe, linewidth=1.5)

    
    plt.title("Categories that were statistically \n significantly different between"+class1+" and "+class2+" subjects")
    # plt.savefig('psi-nsi-most-distinct-boxplot.png')
    plt.show()

    plt.close()

higher_ls = ["Sports","Health","Finance","News","Books & Literature","Reference","Law & Government"]
def ls_nls_analysis():
    
    bars = sorted(["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"])

    psi_users_cat_data = np.loadtxt(open('ls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,28)) )
    print psi_users_cat_data.shape

    nsi_users_cat_data = np.loadtxt(open('nls_cat_search_percent_dist_by_users.csv','rb'), delimiter=",", skiprows=1, usecols=(range(1,30)) )
    print nsi_users_cat_data.shape
    print '=====-----====='
    stat_sig_features_scores = {}
    ssd = []

    for index, category in enumerate(bars):
        if category in higher_ls:
            # category, normalized_ls, normalized_nls = normalize(category, psi_users_cat_data[index], nsi_users_cat_data[index])
            metrics = calc_stats(psi_users_cat_data[index],nsi_users_cat_data[index])
            


            # print metrics['t-test-pval']
            

            # print 'min ls',min(psi_users_cat_data[index]), 'max ls',max(psi_users_cat_data[index]), 'min nls', min(nsi_users_cat_data[index]), 'max nls', max(nsi_users_cat_data[index])
            # print 'min ls',min(psi_users_cat_data[index]), 'min nls', min(nsi_users_cat_data[index])
            # print 'max ls',max(psi_users_cat_data[index]), 'max nls', max(nsi_users_cat_data[index])
            print category,'ls-mean',metrics['x-mean'],'nls-mean', metrics['y-mean'], 'ls median', np.median(psi_users_cat_data[index]),'nls median', np.median(nsi_users_cat_data[index]), 'min ls',min(psi_users_cat_data[index]), 'max ls',max(psi_users_cat_data[index]), 'min nls', min(nsi_users_cat_data[index]), 'max nls', max(nsi_users_cat_data[index]), 'p-val', metrics['t-test-pval']
            
            # if metrics['t-test-pval'] <= 0.05:
            #     print category, metrics['t-test-pval']
            #     print psi_users_cat_data[index]
            #     print '-----'
            #     print nsi_users_cat_data[index]
            #     print '=========================='
            #     stat_sig_features_scores[category] = [psi_users_cat_data[index], nsi_users_cat_data[index]]
            #     ssd.append(category)



    print stat_sig_features_scores
    # create_boxplots(stat_sig_features_scores, ssd, 'LS', 'NLS')
    print ssd
# createHeatmap()

ls_nls_analysis()