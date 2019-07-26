import glob
import json
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

cat_order = ["Business & Industrial","Home & Garden","Travel","Arts & Entertainment","Sports","Food & Drink","Pets & Animals","Health","Shopping","Finance","Adult","Beauty & Fitness","News","Books & Literature","Online Communities","Law & Government","Sensitive Subjects","Science","Hobbies & Leisure","Games","Jobs & Education","Autos & Vehicles","Computers & Electronics","People & Society","Reference","Internet & Telecom","Real Estate"]

higher_ls = ["Sports","Health","Finance","News","Books & Literature","Reference","Law & Government"]
# cat_order = set(cat_order) - set(higher_ls)
cat_order = higher_ls

train_lsFiles = glob.glob("train-low-self-esteem/*.json")
train_nlsFiles = glob.glob("train-not-low-self-esteem/*.json")


def normalixe_dic(d):
	total = 0.0

	for k,v in d.iteritems():
		total += v

	for k,v in d.iteritems():
		d[k] = (v/total)*100.0


	return d

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
    	't-test-pval':t_test_p, 
    	'mw-test-pval':mw_p, 
    	'cohens-d':cohens_d,
    	'kalmogorov':(ks,k_test_p)
    }
    return my_stats

ltracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
mtracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
ntracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
etracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
for file in train_nlsFiles:
	with open(file, 'r') as f:

		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]

				time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p') 
				
				if time.hour > 0 and time.hour <= 5:
					ltracker[top_cat] += 1

				if time.hour > 5 and time.hour <= 12:
					mtracker[top_cat] += 1
				# noon
				if time.hour > 12 and time.hour <=18:
					ntracker[top_cat] += 1
				if time.hour > 18 and time.hour <=23:
					etracker[top_cat] += 1



ltracker = normalixe_dic(ltracker)
mtracker = normalixe_dic(mtracker)
ntracker = normalixe_dic(ntracker)
etracker = normalixe_dic(etracker)
nlstotal = {}
for k in cat_order:
	nlstotal[k] = ltracker[k] + mtracker[k] + ntracker[k] + etracker[k]

#######################
lsltracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
lsmtracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
lsntracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
lsetracker = {'Business & Industrial': 0, 'Home & Garden': 0, 'Travel': 0, 'Arts & Entertainment': 0, 'Sports': 0, 'Food & Drink': 0, 'Pets & Animals': 0, 'Health': 0, 'Shopping': 0, 'Finance': 0, 'Adult': 0, 'Beauty & Fitness': 0, 'News': 0, 'Books & Literature': 0, 'Online Communities': 0, 'Law & Government': 0, 'Sensitive Subjects': 0, 'Science': 0, 'Hobbies & Leisure': 0, 'Games': 0, 'Jobs & Education': 0, 'Autos & Vehicles': 0, 'Computers & Electronics': 0, 'People & Society': 0, 'Reference': 0, 'Internet & Telecom': 0, 'Real Estate': 0}
for file in train_lsFiles:
	with open(file, 'r') as f:

		for line in f.readlines():
			data = json.loads(line)
			if data['category'] != []:
				top_cat = data['category'][0][0].split('/')[1]

				time = datetime.datetime.strptime(data['qtime'],'%b %d, %Y, %I:%M:%S %p') 
				
				if time.hour > 0 and time.hour <= 5:
					lsltracker[top_cat] += 1

				if time.hour > 5 and time.hour <= 12:
					lsmtracker[top_cat] += 1
				# noon
				if time.hour > 12 and time.hour <=18:
					lsntracker[top_cat] += 1
				if time.hour > 18 and time.hour <=23:
					lsetracker[top_cat] += 1



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
    	't-test-pval':t_test_p, 
    	'mw-test-pval':mw_p, 
    	'cohens-d':cohens_d,
    	'kalmogorov':(ks,k_test_p)
    }
    return my_stats

import matplotlib.pyplot as plt


lsltracker = normalixe_dic(lsltracker)
lsmtracker = normalixe_dic(lsmtracker)
lsntracker = normalixe_dic(lsntracker)
lsetracker = normalixe_dic(lsetracker)

lstotal = {}
for k in cat_order:
	lstotal[k] = lsltracker[k] + lsmtracker[k] + lsntracker[k] + lsetracker[k]

total_t = range(len(cat_order))
nseq = np.array([ nlstotal[k] for k in cat_order])
lseq = np.array([ lstotal[k] for k in cat_order])

plt.plot(total_t, nseq, '-o',label='NLS',color='green',linewidth=0.9)
plt.plot(total_t, lseq, '-v',label='LS',color='red',linewidth=0.9)
plt.legend()
plt.grid()
# plt.title('Searching behavior breakdown by categories at anytime of the day')
plt.xticks(total_t, cat_order, rotation=90)
plt.xlim(-1, len(total_t)+1)
plt.ylabel("% searches")
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
# plt.show()
plt.close()



for k in cat_order:
	print k

	print lstotal[k]
	print nlstotal[k]
	print '=-=-'
	# print calc_stats(, )

'''

######################
t = range(len(cat_order))

s1 = np.array([ ltracker[k] for k in cat_order])
lss1 = np.array([ lsltracker[k] for k in cat_order])

plt.plot(t, s1, '-o',label='NLS',color='green',linewidth=0.9)
plt.plot(t, lss1, '-v',label='LS',color='red',linewidth=0.9)
plt.legend()
plt.grid()
plt.title('Searching behavior breakdown by categories during 1am - 5am')
plt.xticks(t, cat_order, rotation=90)
plt.ylabel("% searches")
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()
plt.close()




s1 = np.array([ mtracker[k] for k in cat_order])
lss1 = np.array([ lsmtracker[k] for k in cat_order])

plt.plot(t, s1, '-o',label='NLS',color='green',linewidth=0.9)
plt.plot(t, lss1, '-v',label='LS',color='red',linewidth=0.9)
plt.legend()
plt.grid()
plt.title('Searching behavior breakdown by categories during 5am - Noon')
plt.xticks(t, cat_order, rotation=90)
plt.ylabel("% searches")
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()

plt.close()


s1 = np.array([ ntracker[k] for k in cat_order])
lss1 = np.array([ lsntracker[k] for k in cat_order])

plt.plot(t, s1, '-o',label='NLS',color='green',linewidth=0.9)
plt.plot(t, lss1, '-v',label='LS',color='red',linewidth=0.9)
plt.legend()
plt.grid()
plt.ylabel("% searches")
plt.title('Searching behavior breakdown by categories during Noon - 6pm')
plt.xticks(t, cat_order, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()

plt.close()

s1 = np.array([ etracker[k] for k in cat_order])
lss1 = np.array([ lsetracker[k] for k in cat_order])

plt.plot(t, s1, '-o',label='NLS',color='green',linewidth=0.9)
plt.plot(t, lss1, '-v',label='LS',color='red',linewidth=0.9)
plt.legend()
plt.grid()
plt.ylabel("% searches")
plt.title('Searching behavior breakdown by categories during 6pm - Midnight')
plt.xticks(t, cat_order, rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()
'''
'''
############################################
import matplotlib.pyplot as plt
# import numpy as np

# t = np.arange(0.01, 5.0, 0.01)
s1 = np.array([ ltracker[k] for k in cat_order])
s2 = np.array([ mtracker[k] for k in cat_order])
s3 = np.array([ ntracker[k] for k in cat_order])
s4 = np.array([ etracker[k] for k in cat_order])

# ax1 = plt.subplot(411)
plt.plot(t, s1, '-o',label='1am - 5pm',color='red',linewidth=0.9)
# plt.legend()
# plt.grid()
# plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
# ax2 = plt.subplot(412, sharex=ax1)
plt.plot(t, s2,'-v',label='5am - Noon',color='green',linewidth=0.9)
# plt.legend()
# plt.grid()
# make these tick labels invisible
# plt.setp(ax2.get_xticklabels(), visible=False)

# # # share x and y
# ax3 = plt.subplot(413, sharex=ax1, sharey=ax1)
plt.plot(t, s3,'-1',label='Noon - 6pm',color='purple',linewidth=0.9)
# plt.legend()
# plt.grid()
# plt.xticks(t, cat_order, rotation=90)

# plt.setp(ax2.get_xticklabels(), visible=False)

# ax4 = plt.subplot(414, sharex=ax1, sharey=ax1)
plt.plot(t,s4,'-*',label='6pm - Midnight',color='orange',linewidth=0.9)
# plt.legend()
# plt.grid()
# plt.xticks(t, cat_order, rotation=90)

# plt.setp(ax2.get_xticklabels(), visible=False)

plt.xticks(t, cat_order, rotation=90)
plt.legend()
plt.grid()
plt.title('NLS:Searching behavior breakdown by categories')
plt.ylabel("% of searchers")

plt.show()
'''