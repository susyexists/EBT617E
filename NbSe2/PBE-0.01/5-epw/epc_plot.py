import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import sys
data = np.loadtxt('NbSe2.freq.gp')
symmetryfile = 'plotband.out'
lbd = np.loadtxt("lambda.dat")
lbd_val = np.where(lbd<1 , lbd, 1)
def Symmetries(fstring):
    f = open(fstring, 'r')
    x = np.zeros(0)
    for i in f:
        if "high-symmetry" in i:
            x = np.append(x, float(i.split()[-1]))
    f.close()
    return x
x=np.tile(data.T[0],9)
val = lbd_val.T[1:].reshape(-1)
y=data.T[1:].reshape(-1,)
fig=plt.figure(figsize=(8,6))
labels=["G","M","K","G"]
plt.scatter(x,y*0.12398,c=val,cmap="copper",s=10)
sym_tick = Symmetries(symmetryfile)
for i in range(len(sym_tick)-1):
    plt.axvline(sym_tick[i],linestyle='dashed', color='black', alpha=0.75)
plt.xticks(sym_tick,labels)
plt.xlim(min(sym_tick),max(sym_tick))
plt.ylim(0)
plt.ylabel("Energy (meV)")
plt.colorbar()
plt.savefig("epc.pdf")
plt.show()