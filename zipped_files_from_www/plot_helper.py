import numpy as np
import matplotlib.pyplot as plt
from helper import *
import sys
sys.path.insert(0,'../Boyu')
from all_cls_result_lists import *
#features = ['Lambda','Pretrained','Lambda+Self_Esteem']
 
method_index = [0,1,2,7]

required_methods = required_list(methods,method_index)

fig, ax = plt.subplots()
xaxis = np.arange(len(required_methods))
#nethods = np.arange(n)
bar_width = 0.2

m1 = required_list(lambda_results,method_index)
m2 = required_list(cat_results,method_index)
m3 = required_list(pretrained_results,method_index)
m4 = required_list(lambda_and_se_results,method_index)

#print(m1)

r1 = ax.bar(xaxis, m1, bar_width,label = features[0])

r2 = ax.bar(xaxis + bar_width, m2, bar_width,label = features[1])

r3 = ax.bar(xaxis + bar_width+ bar_width, m3, bar_width,label = features[2])

r4 = ax.bar(xaxis + bar_width+ bar_width+ bar_width, m4, bar_width, label = features[3])

ax.set_ylim([0, 1])

# r1 = ax.barh(xaxis, m1, bar_width,label = features[0])

# r2 = ax.barh(xaxis + bar_width, m2, bar_width,label = features[1])

# r3 = ax.barh(xaxis + bar_width+ bar_width, m3, bar_width,label = features[2])

# r4 = ax.barh(xaxis + bar_width+ bar_width+ bar_width, m4, bar_width, label = features[3])

# ax.set_ylim([0, 1])

ax.set_ylabel('Average F1 Score',fontsize=16)
ax.set_title('F1 Score for different classifiers Using different features',fontsize=16)
ax.set_xticks(xaxis+1.5*bar_width)
ax.set_xticklabels(required_methods,fontsize=16)
ax.text(2,0.8,'Lambda Feature is stronger')
ax.legend()

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

# ax1.bar(x,m1, 0.2) #% thickness=0.2
# ax2.bar(x,m2, 0.2)
# ax3.plot(x,m3)
# ax4.plot(x,m4)

plt.tight_layout()
plt.show()