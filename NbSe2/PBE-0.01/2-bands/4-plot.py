#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import sys


datafile = 'bands.dat.gnu'
data = np.loadtxt(datafile)
fermi =  0.2429
symmetryfile = '3-bands_pp.out'
labels=["G","M","K","G"]


# This function extracts the high symmetry points from the output of bandx.out
def Symmetries(fstring):
    f = open(fstring, 'r')
    x = np.zeros(0)
    for i in f:
        if "high-symmetry" in i:
            x = np.append(x, float(i.split()[-1]))
    f.close()
    return x

x=data.T[0]
y=data.T[1:]-fermi

fig=plt.figure(figsize=(8,6))

for i in y:
    plt.scatter(x,i,s=1,c="black")
sym_tick = Symmetries(symmetryfile)
for i in range(len(sym_tick)-1):
    plt.axvline(sym_tick[i],linestyle='dashed', color='black', alpha=0.75)
plt.axhline(0,linestyle='dashed', color='red', alpha=0.75)
plt.xticks(sym_tick,labels)
plt.xlim(min(sym_tick),max(sym_tick))
plt.ylim(-3,3)
plt.ylabel("Energy (eV)")
plt.savefig("band.png")
plt.show()
