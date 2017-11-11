import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np


alloc = np.genfromtxt('../../data/alloc2.csv', delimiter=' ', names=['size', 'cuda', 'numa', 'memset'])


fig, ax1 = plt.subplots()
ind = np.arange(len(alloc['size']))
w = 0.2
#int = [128, 256, 512, 1024, 2048]

#ax1.set_title("Allocation Time")    

ax1.set_yscale('log')

ax1.set_xticks(ind)
ax1.set_xticklabels(['128','256','512','1024','2048'])
ax1.set_ylabel('Time[us]')
ax1.set_xlabel('size[Mbyte]')
    

r3 = ax1.bar(ind+w, alloc['numa'], w, color='b', label='numaMallocManaged')
r2 = ax1.bar(ind, alloc['memset'],w, color='y', label='cudaMallocManaged + memset')
r1 = ax1.bar(ind-w, alloc['cuda'], w, color='g', label='cudaMallocManaged')

leg = ax1.legend()

plt.show()
