# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# cut_bin = 4
# orig = [80,90,100,150,300,250,1600,230,200,210,170,400,-800,500,530,550]
# orig1 = orig[:]
# orig2 = orig[:]
#
# fig = plt.figure(figsize=(16,16),dpi=66)
# ax1 = fig.add_subplot(111)
# x = [i for i in range(len(orig))]
# ax1.plot(x,orig,label='原始数据')
#
# orig_sort1 = sorted(orig1)
# id1 = []
# for i in orig_sort1: id1.append(orig1.index(i))
# for i in range(0,len(orig1),cut_bin): orig_sort1[i:i+cut_bin] = [np.mean(orig_sort1[i:i+cut_bin])]*cut_bin
# j = 0
# for i in id1:
#     orig1[i] = orig_sort1[j]
#     j += 1
# ax1.plot(x,orig1,label='均值平滑')
#
# orig_sort2 = sorted(orig2)
# print(orig_sort2)
# id2=[]
# for i in orig_sort2:
#     id2.append(orig2.index(i))
# for i in range(0,len(orig1),cut_bin):
#     orig_sort2[i:i+cut_bin] =  [np.median(orig_sort2[i:i+cut_bin])]*cut_bin
# j = 0
# for i in id2:
#     orig2[i] = orig_sort2[j]
#     j += 1
# print(orig2)
# ax1.plot(x,orig2,label='中值平滑')
# ax1.legend(loc="upper right")
# fig.text(0.5,0.5,'等深分箱',fontsize=80,alpha=0.2,va='center',ha='center')
# ax1.set_title("原始数据,均值滑动,中值滑动对比图")
# plt.show()
#
# # 边界平滑
# for i in orig_sort2:
#     id2.append(orig2.index(i))
# for i in range(1,len(orig2),cut_bin):
#     if orig_sort2[i+2]-orig_sort2[i] < orig_sort2[i]-orig_sort2[i-1]: orig_sort2[i] = orig_sort2[i + 2]
#     else: orig_sort2[i] = orig_sort2[i - 1]
#     if orig_sort2[i+2]-orig_sort2[i+1] < orig_sort2[i+1]-orig_sort2[i-1]: orig_sort2[i+1] = orig_sort2[i + 2]
#     else: orig_sort2[i+1] = orig_sort2[i - 1]

import numpy as np
import pdd as pd
import matplotlib.pyplot as plt

cut_bin = 4
orig = [80,90,100,150,300,250,1600,230,200,210,170,400,-800,500,530,550]
bin_size = (max(orig)-min(orig))/cut_bin
orig1 = orig[:]
orig2 = orig[:]

fig = plt.figure(figsize=(16,16),dpi=66)
ax1 = fig.add_subplot(111)
x = [i for i in range(len(orig))]
ax1.plot(x,orig,label='原始数据')

orig_sort1 = sorted(orig1)
id1 = []
for i in orig_sort1:
    id1.append(orig1.index(i))
bin = [[] for i in range(4)]
j = 0
for i in range(0,len(orig1)):
    if orig_sort1[i] < min(orig_sort1) + (j+1) * bin_size:
        bin[j].append(orig_sort1[i])
    else:
        j += 1
        bin[j].append(orig_sort1[i])
for i in range(4):
    bin[i] = [np.mean(bin[i])]*len(bin[i])

orig_sort1=[]
for i in range(len(bin)):
    for j in range(len(bin[i])):
        orig_sort1.append(bin[i][j])
j = 0
for i in id1:
    orig1[i] = orig_sort1[j]
    j += 1
ax1.plot(x,orig1,'o',label='均值平滑',)
print(orig1)

bin2 = [[] for i in range(4)]
orig_sort2 = sorted(orig2)
id2=[]
for i in orig_sort2:
    id2.append(orig2.index(i))
j=0
for i in range(0,len(orig2)):
    if orig_sort2[i] < min(orig_sort2) + (j+1) * bin_size:
         bin2[j].append(orig_sort2[i])
    else:
        j += 1
        bin2[j].append(orig_sort2[i])
for i in range(4):
    bin2[i] = [np.median(bin2[i])]*len(bin2[i])
orig_sort2=[]
for i in range(len(bin)):
    for j in range(len(bin[i])):
        orig_sort2.append(bin[i][j])
j = 0
for i in id2:
    orig2[i] = orig_sort2[j]
    j += 1
ax1.plot(x,orig2,'r--*',label='中值平滑')
ax1.legend(loc="upper right")
ax1.set_title("原始数据,均值滑动,中值滑动对比图")
fig.text(0.5,0.5,'等宽分箱',fontsize=80,alpha=0.2,va='center',ha='center')
plt.show()