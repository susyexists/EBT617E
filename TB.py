from numpy import * 
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()-4

class TB:
    
    def __init__(self, data, nbnd, points, fe,super_cell=1):  
        
        self.hopping = data[0]
        self.nbnd = nbnd
        self.points = points
        self.fe = fe
        self.super_cell = super_cell  
        self.sym = data[2]
        self.h = self.hopping.reshape(self.points,self.nbnd*self.nbnd)
        self.x = data[1].reshape(2,points,nbnd*nbnd)
        
        
    def fourier (self,k):
        kx = tensordot(k,self.x,axes=(0,0))
        transform =dot(self.sym,exp(-1j*self.super_cell*kx)*self.h).reshape(self.nbnd,self.nbnd)
        return(transform)


    def eig(self,k):
        val = []
        vec = []
        for i in range(len(k)):
            sol  = linalg.eigh(self.fourier(k[i]))
            val.append(sol[0])
            vec.append(sol[1])
        return (val,vec)    
    def bands(self):
               
        n=101
        l=117
        m=275-217
        mg=array([zeros(n),linspace(3.1,0,n)]).T
        gk=array([linspace(0,2.1,l),linspace(0,2.1,l)]).T
        km=array([linspace(2.1,0,m),-1/2.1*linspace(2.1,0,m)+3.1]).T
        path=concatenate([mg,gk,km])
    
        val, vec = self.eig(path)
        plt.figure(figsize=(8,8))
        for i in range(self.nbnd):
            plt.plot(array(val).T[i]-self.fe,c="black")
        plt.xticks(ticks=[0,len(mg),len(mg)+len(gk),len(mg)+len(gk)+len(km)],labels=["M","G","K","M"])
        plt.plot([0,0],[plt.ylim()[0],plt.ylim()[1]],color="black")
        plt.plot([len(mg),len(mg)],[plt.ylim()[0],plt.ylim()[1]],color="black")
        plt.plot([len(mg)+len(gk),len(mg)+len(gk)],[plt.ylim()[0],plt.ylim()[1]],color="black")
        plt.plot([len(mg)+len(gk)+len(km),len(mg)+len(gk)+len(km)],[plt.ylim()[0],plt.ylim()[1]],color="black")
        plt.plot([plt.xlim()[0],plt.xlim()[1]],[0,0],color="black")
        plt.xlim(0,len(path))
        plt.ylim(-3,3)
#         plt.show()
    def solver (self,k):
        kx = tensordot(k,self.x,axes=(0,0))
        transform =dot(self.sym,exp(-1j*self.super_cell*kx)*self.h).reshape(self.nbnd,self.nbnd)
        val, vec = linalg.eigh(transform)
        return(val)
    
    def parallel_solver(self,path):
        results = Parallel(n_jobs=num_cores)(delayed(self.solver)(i) for i in path)
        res = array(results).T-self.fe
        return (res)
    
    def fermi(self,e):
        return where(e>0,0,1)
    
    def suscep(self,point,mesh,mesh_energy,mesh_fermi):
#         xk = self.parallel_solver(mesh)[6]
#         kf = self.fermi(mesh_energy
        shifted_energy = self.parallel_solver(point+mesh)[6]
        shifted_fermi = self.fermi(shifted_energy)
        num = mesh_fermi-shifted_fermi
        den = mesh_energy-shifted_energy
        res = average(num/den)
        return(append(point,res))
    
    def hexagon():
        a = array([[[-1/sqrt(3),1/sqrt(3)],[1,1]],
        [[1/sqrt(3),2/sqrt(3)],[1,0]],
        [[2/sqrt(3),1/sqrt(3)],[0,-1]],
        [[1/sqrt(3),-1/sqrt(3)],[-1,-1]],
        [[-1/sqrt(3),-2/sqrt(3)],[-1,0]],
        [[-2/sqrt(3),-1/sqrt(3)],[0,1]],
        ])
        return (a)
    
    def plot_path(self,band):
        n=101
        l=117
        m=275-217
        mg=array([zeros(n),linspace(3.1,0,n)]).T
        gk=array([linspace(0,2.1,l),linspace(0,2.1,l)]).T
        km=array([linspace(2.1,0,m),-1/2.1*linspace(2.1,0,m)+3.1]).T
        path=concatenate([mg,gk,km])
        plt.figure(figsize=(8,8))
        for i in range(self.nbnd):
            plt.plot(band[i] - self.fe,c="white")     
        plt.xticks(ticks=[0,len(mg),len(mg)+len(gk),len(mg)+len(gk)+len(km)],labels=["M","G","K","M"])
        plt.plot([0,0],[plt.ylim()[0],plt.ylim()[1]],color="white")
        plt.plot([len(mg),len(mg)],[plt.ylim()[0],plt.ylim()[1]],color="white")
        plt.plot([len(mg)+len(gk),len(mg)+len(gk)],[plt.ylim()[0],plt.ylim()[1]],color="white")
        plt.plot([len(mg)+len(gk)+len(km),len(mg)+len(gk)+len(km)],[plt.ylim()[0],plt.ylim()[1]],color="white")
        plt.plot([plt.xlim()[0],plt.xlim()[1]],[0,0],color="white")
        plt.xlim(0,len(path))
        # plt.legend()
        plt.ylim(-0.5,1)
        plt.show()
        
def read_hr(hr,sym):
    sym = loadtxt('{}.dat'.format(sym),dtype=int)
    wannier= loadtxt('{}.dat'.format(hr)).T
    x = wannier[0:2]
#     index = wannier[3:5].T-1
    hopping = wannier[5]+1j*wannier[6]
    return (hopping,x,sym)