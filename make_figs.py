import os

import numpy as np
import matplotlib.pyplot as plt

import ising_sw

np.random.seed(seed=0) # Reproducibility

def make_warmup_time_figure():
    out_path = os.path.join('Figures/Warmup_Example.eps')
    size = 32
    temperature = 1.13459265711 # This is the critical temperature for our thing
    nsteps = 100000
    lat = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
    e, m = ising_sw.ising_simulation(size, temperature, nsteps, lat)
    e /= size**2
    plt.figure(figsize=(5,4))
    plt.plot(e)
    plt.xlabel('Time-Step')
    plt.ylabel('Energy per Atom')
    plt.savefig(out_path)


if __name__ == '__main__':
    make_warmup_time_figure()