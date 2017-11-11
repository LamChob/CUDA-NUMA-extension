import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

app_data = np.genfromtxt('../../data/p1a1-3.csv', delimiter=' ', names=['size', 'nit', 'nt', 'alloc', 'th2h', 't1', 't2', 'h2d1', 'd2h1', 'h2d2','d2h2', 'd2n'])

h2h = np.genfromtxt('../../data/bw.csv', delimiter=' ', names=['size', 'h2h', 'n2d'])

fig = plt.figure()

ind = np.arange(5)
w = 0.15

ax1 = fig.add_subplot(111)
#ax1.set_title("Bandwidth")    
ax1.set_xlabel('size[Mbyte]')
ax1.set_ylabel('GByte/s')

ax1.set_xticks(ind)
ax1.set_xticklabels(['128','256','512','1024','2048'])

#h2h = np.empty([5], dtype=float)
#d2h = np.empty([5], dtype=float)
#h2d = np.empty([5], dtype=float)

d2n = []
d2h = []
h2d = []
step = 49
r = np.arange(0,244,step)
for x in r:
    tmp = np.array(app_data)[x:x+49:1]
    d2n = np.append(d2n, np.average(tmp['d2n']))
    d2h = np.append(d2h, np.average(tmp['d2h1']))
    h2d = np.append(h2d, np.average(tmp['h2d1']))

r1 = ax1.bar(ind+w*2, h2d, w, color='g', label='Host-to-Device')
r2 = ax1.bar(ind+w, d2h, w, color='y', label='Device-to-Host')
r2 = ax1.bar(ind, h2h['n2d'], w, color='black', label='Node-to-Device')
r4 = ax1.bar(ind-w, h2h['h2h'], w, color='r', label='Host-to-Host')
r3 = ax1.bar(ind-w*2, d2n, w, color='b', label='Device-to-Node')

leg = ax1.legend()

plt.show()
