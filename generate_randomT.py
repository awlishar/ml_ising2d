import os
import numpy as np
import matplotlib.pyplot as plt
import time
from ising import IsingModel2D, plotConfigs
import random

##############################################################################
start_time = time.time()
folder = "data/"

onsagerIsingTc = 1/(1/2*np.log(1+np.sqrt(2))) # Exact solution 2.269185314213022
temperatures = []
nt = 100
t_range = 2.0

energies = np.zeros(nt)
magnetizations = np.zeros(nt)
mcsteps = int(1e4)
i = 0
N = 16
print(f"N={N}, Number of sites={N*N}, Monte Carlo steps={mcsteps}")
np.save(folder + 'meta', [N, mcsteps, t_range])
confs = []

datalist = []
elist = []
mlist = []
clist = []

for i in range(0,nt):
    print(i)
    t = random.uniform(onsagerIsingTc-t_range/2, onsagerIsingTc+t_range/2)
    temperatures.append(t)
    ising = IsingModel2D(N=N, T=t)
    ising.runMonteCarlo2(mcsteps)
    energies[i] = ising.energy
    magnetizations[i] = ising.magnetization
    confs.append(ising)
    datalist.append(ising.lattice)
    elist.append(ising.energy)
    mlist.append(ising.magnetization)
    clist.append(0 if ising.T < onsagerIsingTc else 1)
print("--- %s seconds ---" % (time.time() - start_time))


np.save(folder + 'data', datalist)
np.save(folder + 'elist', elist)
np.save(folder + 'mlist', mlist)
np.save(folder + 'clist', clist)
np.save(folder + 'tlist', temperatures)
