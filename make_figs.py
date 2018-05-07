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
    e, __ = ising_sw.ising_simulation(size, temperature, nsteps, lat)
    e /= size**2
    plt.figure(figsize=(5,4))
    plt.plot(e)
    plt.xlabel('Time-Step')
    plt.ylabel('Energy per Atom')
    plt.savefig(out_path)


def make_pcolor_figures():
    size = 100
    tc = 1.13459265711
    t1 = 0.5 * tc
    t2 = 1.5 * tc
    base = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
    # WHY DOESN'T THIS WORK OMGGGGGGGGG

    lat = np.copy(base)
    # Set the periodic bc's
    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
    plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_1_Tc.eps'))

    for __ in range(10000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, tc)
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_10000_Tc.eps'))

    for __ in range(90000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, tc)
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_100000_Tc.eps'))

    lat = np.copy(base)
    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_1_0.5Tc.eps'))
    for __ in range(10000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, t1)
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_10000_0.5Tc.eps'))
    for __ in range(90000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, t1)
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_100000_0.5Tc.eps'))

    lat = np.copy(base)
    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
    #plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_1_1.5Tc.eps'))
    for __ in range(10000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, t2)
    plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_10000_1.5Tc.eps'))
    for __ in range(90000):
        print('simming')
        lat, __ = ising_sw.ising_step(lat, size, t2)
    plt.figure(figsize=(4,4))
    plt.pcolor(lat[1:-1, 1:-1])
    plt.xlabel('Does this work?')
    plt.savefig(os.path.join('Figures/Step_100000_1.5Tc.eps'))

if __name__ == '__main__':
    make_pcolor_figures()
    #make_warmup_time_figure()