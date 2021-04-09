import numpy as np
import matplotlib.pyplot as plt
ef = 6.5701
dos=np.loadtxt("Si.dos.dat")
dos[:,0]-=ef
plt.plot(dos[:,0],dos[:,1],"--")
plt.xlim(-12,15)
plt.show()
