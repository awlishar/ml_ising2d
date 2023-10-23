import numpy as np
import matplotlib.pyplot as plt
import time
# from numba import jit
# from numba.experimental import jitclass
# from numba import int32, float64

from matplotlib import colors


class IsingModel2D:
    def __init__(self, N: int = 4, T: float = 0.0, J: float = 1.0):
        # Square lattice N x N
        self.N = N
        # Temperature in K
        self.T = T
        self.J = J
        # Generate a random lattice with either 1 or -1
        self.lattice = np.random.choice([1, -1], size=(N, N))
        # Energy of the initial lattice
        self.energy = self.getEnergy()
        self.magnetization = self.getMag()

    def __str__(self):
        return f"N={self.N} \nT={self.T}K \nE={self.energy} \nM={self.magnetization} \n{self.lattice} "

    def getMag(self):
        return np.sum(self.lattice)

    def getEnergy(self):
        ''''''
        E = 0
        J = self.J
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    sigma_ij = self.lattice[i, j]
                    # Nearest neighbours
                    sigma_sum_nn = (
                        self.lattice[(i - 1) % self.N, j]
                        + self.lattice[(i + 1) % self.N, j]
                        + self.lattice[i, (j - 1) % self.N]
                        + self.lattice[i, (j + 1) % self.N])
                    E += -J * sigma_ij * sigma_sum_nn
        return E/4

    def getdE(self, i: int = 0, j: int = 0):
        # Calculate change in energy
        sigma_ij = self.lattice[i, j]
        # Nearest neighbours
        sigma_sum_nn = (
            self.lattice[(i - 1) % self.N, j]
            + self.lattice[(i + 1) % self.N, j]
            + self.lattice[i, (j - 1) % self.N]
            + self.lattice[i, (j + 1) % self.N])

        dE = 2 * sigma_ij * sigma_sum_nn
        return dE

    def runMonteCarlo2(self, mcsteps: int = 1000):
        # Monte Carlo simulation: Metropolis algorithm
        beta = 1/self.T
        self.energy = self.getEnergy()
        self.magnetization = 0.0

        for step in range(mcsteps):
            # print(f"step={step}")
            for x in range(self.N*self.N):

                i = np.random.randint(0, self.N)
                j = np.random.randint(0, self.N)

                dE = self.getdE(i, j)

                if dE < 0 or np.random.random() < np.exp(-beta*dE):
                    # Switch spin at i,j
                    # print(self.energy, dE)
                    self.lattice[i, j] = -self.lattice[i, j]
                # self.energy += dE
            # print(self.energy)

            self.energy += self.getEnergy()
            self.magnetization += self.getMag()

        return self.lattice

# visualize configurations
def plotConfigs(confs):
    fig, axes = plt.subplots(1,len(confs))
    cmap = colors.ListedColormap(['yellow', 'purple'])
    bounds=[-1,0,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #fig.figsize(10, len(confs)*10)
    for i in range(len(confs)):
        ax = axes[i]
        config = confs[i]
        img = ax.imshow(config.lattice, interpolation=None, cmap=cmap, norm=norm)
        # make a color bar
        # ax.set_title(f"T={config.T:.2f}K, E={config.energy}, M={config.magnetization}")

    # fig.colorbar(img, boundaries=bounds, ticks=[-1, 1])
    plt.show()
