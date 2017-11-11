import matplotlib as mpl
import sys
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

file1 = sys.argv[1]
file2 = sys.argv[2]
data0 = np.genfromtxt(file1, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])

data1 = np.genfromtxt(file2, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Short Kernels")
ax1.set_xlabel('workload size (MByte)')
ax1.set_ylabel('Increased runtime(%)')

#d = np.array(app_data)[0:225:15]
#e = np.array(app_data)[14:225:15]

a0hi = []
a0li = []
a1hi = []
a1li = []

# offsets inside chunk
lt = 1 # 3 movements
li = 7 # 3 iterations
ht = 6 # 16 movements
hi = 42 # 16 iterations

step = 49
r = np.arange(0,len(data1),step)
for x in r:
    # relative difference with affinity (p1a1)
    slice = np.array(data0)[x:x+step:1]
    distance = np.subtract(slice['t2'], slice['t1'])
    factor = np.divide(distance, slice['t1'])
    a0li = np.append(a0li, factor[lt+li]*100)
    a0hi = np.append(a0hi, factor[ht+li]*100)

    slice = np.array(data1)[x:x+step:1]
    distance = np.subtract(slice['t2'], slice['t1'])
    factor = np.divide(distance, slice['t1'])
    a1li = np.append(a1li, factor[lt+li]*100)
    a1hi = np.append(a1hi, factor[ht+li]*100)

xaxis = [128, 256, 512, 1024, 2048]
ind = np.arange(5)
w = 0.2

#ax1.set_xscale('log')
ax1.set_xticklabels(['128','256','512','1024','2048'])
ax1.set_xticks(ind)

ax1.bar(ind-w, a0li,w, color='black', label='No affinity, low datamovement')
ax1.bar(ind, a1li,w, color='y', label='Affinity, low datamovement')
ax1.bar(ind+w, a0hi,w,color='blue', label='No affinity, high datamovement')
ax1.bar(ind+2*w, a1hi,w, color='g', label='Affinity, high datamovement')


leg = ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_title("Long Kernels")
ax2.set_xlabel('workload size (MByte)')
ax2.set_ylabel('Increased runtime(%)')

#d = np.array(app_data)[0:225:15]
#e = np.array(app_data)[14:225:15]

a0hi = []
a0li = []
a1hi = []
a1li = []

# offsets inside chunk
lt = 1 # 3 movements
li = 7 # 3 iterations
ht = 6 # 16 movements
hi = 42 # 16 iterations

step = 49
r = np.arange(0,len(data1),step)
for x in r:
    # relative difference with affinity (p1a1)
    slice = np.array(data0)[x:x+step:1]
    distance = np.subtract(slice['t2'], slice['t1'])
    factor = np.divide(distance, slice['t1'])
    a0li = np.append(a0li, factor[lt+hi]*100)
    a0hi = np.append(a0hi, factor[ht+hi]*100)

    slice = np.array(data1)[x:x+step:1]
    distance = np.subtract(slice['t2'], slice['t1'])
    factor = np.divide(distance, slice['t1'])
    a1li = np.append(a1li, factor[lt+hi]*100)
    a1hi = np.append(a1hi, factor[ht+hi]*100)

xaxis = [128, 256, 512, 1024, 2048]
ind = np.arange(5)
w = 0.2

#ax2.set_xscale('log')
ax2.set_xticklabels(['128','256','512','1024','2048'])
ax2.set_xticks(ind)

ax2.bar(ind-w, a0li,w, color='black', label='No affinity, low datamovement')
ax2.bar(ind, a1li,w, color='y', label='Affinity, low datamovement')
ax2.bar(ind+w, a0hi,w,color='blue', label='No affinity, high datamovement')
ax2.bar(ind+2*w, a1hi,w, color='g', label='Affinity, high datamovement')


leg = ax2.legend()

plt.show()
