import numpy as np
import matplotlib.pyplot as plt
from helper import *
import sys
sys.path.insert(0,'../Boyu')
from all_cls_result_lists import *
#features = ['Lambda','Pretrained','Lambda+Self_Esteem']
 
method_index = [0,1,2,3,4]
#print(len(methods))
required_methods = required_list(methods,method_index)
#print(required_methods,"hi")
fig, ax = plt.subplots()
xaxis = np.arange(5)
#nethods = np.arange(n)
bar_width = 0.2

m1, m1_error = cat_results, cat_ci#required_list(lambda_results,method_index)
m2, m2_error = lambda_results, lambda_ci#required_list(cat_results,method_index)
m3, m3_error = lambda_and_se_results, lambda_and_se_ci#required_list(pretrained_results,method_index)
#m4 = required_list(lambda_and_se_results,method_index)





# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(True)


r1 = ax.bar(xaxis, m1,  bar_width,yerr=m1_error,label = features[0],color = '#a6611a',capsize=3)

r2 = ax.bar(xaxis + bar_width,m2,bar_width,yerr=m2_error, label = features[1],color = '#dfc27d',capsize=3)

r3 = ax.bar(xaxis + bar_width+ bar_width,m3, bar_width, yerr=m3_error,label = features[2],color = '#80cdc1',capsize=3)

#r4 = ax.bar(xaxis + bar_width+ bar_width+ bar_width, m4, bar_width, label = features[3],color = '#018571')

ax.set_ylim([0.4, 0.85])

# r1 = ax.barh(xaxis, m1, bar_width,label = features[0])

# r2 = ax.barh(xaxis + bar_width, m2, bar_width,label = features[1])

# r3 = ax.barh(xaxis + bar_width+ bar_width, m3, bar_width,label = features[2])

# r4 = ax.barh(xaxis + bar_width+ bar_width+ bar_width, m4, bar_width, label = features[3])

# ax.set_ylim([0, 1])

ax.set_ylabel('Average F1 Score',fontsize=24)
#ax.set_title('F1 Score for different classifiers Using different features',fontsize=16)
ax.set_xticks(xaxis+1.5*bar_width)
ax.set_xticklabels(required_methods,fontsize=20)
ax.axhline(0.76,color='#dfc27d',linewidth=3,linestyle='dashed')
#ax.text(2,0.8,'Lambda Feature is stronger')
ax.annotate('Graphical model avg F1 score (0.76)',xy=(1.2,0.762),xytext=(1.5,0.8),fontsize=18,arrowprops= dict(facecolor='black',shrink=0.05))
ax.legend(loc='lower right',fontsize=20)
plt.yticks(fontsize=24)
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

# ax1.bar(x,m1, 0.2) #% thickness=0.2
# ax2.bar(x,m2, 0.2)
# ax3.plot(x,m3)
# ax4.plot(x,m4)

plt.tight_layout()
plt.show()