import matplotlib as mpl
import sys
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]

data0 = np.genfromtxt(file1, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])

data1 = np.genfromtxt(file2, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])
fig = plt.figure()

data3 = np.genfromtxt(file1, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])

data4 = np.genfromtxt(file2, delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'td2n', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_title("with NUMA affinity")
ax1.set_xlabel('workload size (MByte)')
ax1.set_ylabel('total runtime differencd')

#d = np.array(app_data)[0:225:15]
#e = np.array(app_data)[14:225:15]

a0hi = []
a0li = []
a1hi = []
a1li = []

# offsets inside chunk
lt = 0 # 3 movements
li = 0 # 3 iterations
ht = 1 # 16 movements
hi = 2 # 16 iterations

step = 4
r = np.arange(0,15,step)
for x in r:
    # relative difference with affinity (p1a1)
    slice1 = np.array(data0)[x:x+step:1]
    slice2 = np.array(data1)[x:x+step:1]
    sum1 = np.add(slice1['t2'], slice['t1'])
    sum2 = np.add(slice1['t2'], slice['t1'])
    distance = np.subtract(sum2, sum1)
    a0li = np.append(a0li, distance[lt+li])
    a0hi = np.append(a0hi, distance[ht+li])
    a1li = np.append(a1li, distance[lt+hi])
    a1hi = np.append(a1hi, distance[ht+hi])

xaxis = [128, 256, 512, 1024]
ind = np.arange(4)
w = 0.2

#ax1.set_xscale('log')
ax1.set_xticklabels(['128','256','512','1024','2048'])
ax1.set_xticks(ind)

ax1.bar(ind-w, a0li,w, color='black', label='No affinity, 3 iterations')
ax1.bar(ind, a1li,w, color='y', label='Affinity, 3 iterations')
ax1.bar(ind+w, a0hi,w,color='blue', label='No affinits, 16 iterations')
ax1.bar(ind+2*w, a1hi,w, color='g', label='Affinity, 16 iterations')

leg = ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_title("16 kernel iterations")
ax2.set_xlabel('workload size (MByte)')
ax2.set_ylabel('Increased runtime(%)')

#d = np.array(app_data)[0:225:15]
#e = np.array(app_data)[14:225:15]

a0hi = []
a0li = []
a1hi = []
a1li = []

r = np.arange(0,15,step)

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

xaxis = [128, 256, 512, 1024]
ind = np.arange(4)
w = 0.2

#ax2.set_xscale('log')
ax2.set_xticklabels(['128','256','512','1024','2048'])
ax2.set_xticks(ind)

ax2.bar(ind-w, a0li,w, color='black', label='No affinity, 3 iterations')
ax2.bar(ind, a1li,w, color='y', label='Affinity, 3 iterations')
ax2.bar(ind+w, a0hi,w,color='blue', label='No affinits, 16 iterations')
ax2.bar(ind+2*w, a1hi,w, color='g', label='Affinity, 16 iterations')


leg = ax2.legend()

plt.show()
