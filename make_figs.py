import os

import numpy as np
import matplotlib.pyplot as plt

import ising_sw

#np.random.seed(seed=0) # Reproducibility

def make_warmup_time_figure():
    out_path = os.path.join('Figures/Warmup_Example.eps')
    size = 32
    temperature = 1.13459265711 # This is the critical temperature for our thing
    nsteps = 100000
    lat = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
    e, __ = ising_sw.ising_simulation(size, temperature, nsteps, lat)
    e /= size**2
    plt.figure(figsize=(5,4))
    plt.plot(e)
    plt.xlabel('Time-Step')
    plt.ylabel('Energy per Atom')
    plt.show()
    #plt.savefig(out_path)


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


def make_acvf_figures():
    # Okay i need to create like 100 sims and average them
    size = 32
    nsims = 100
    nsteps = 20000
    tc = 3
    en_arr = np.zeros(shape=(nsims, nsteps+1))
    print('*********************************************')
    print('STARTING ACVF FIGURES. THIS TAKES A LONG TIME')
    print('*********************************************')

    for i in range(nsims):
        #np.random.seed(seed=i)
        lat = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
        lat[ 0,  1:-1] = lat[-2,  1:-1]
        lat[-1,  1:-1] = lat[ 1,  1:-1]
        lat[ 1:-1,  0] = lat[ 1:-1, -2]
        lat[ 1:-1, -1] = lat[ 1:-1,  1]
        lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
        en_arr[i, :], __ = ising_sw.ising_simulation(size, tc, nsteps, lat)
        print('FINISHED STEP {}'.format(i))

    acvf = []
    for i in range(nsims):
        print('CALC #{}'.format(i))
        acvf.append(ising_sw.compute_ACcF(en_arr[i, :], 10000, 15000, 5000))

    acvf = np.array(acvf)
    av = []
    for i in range(acvf.shape[1]):

        av.append(acvf[:, i].mean())
    plt.figure(figsize=(4,4))
    plt.plot(av)
    plt.xlabel("Simulation Time Shift")
    plt.ylabel("Autocorrelation")
    plt.savefig('Figures/acvf.eps')

if __name__ == '__main__':
    #make_pcolor_figures()
    make_warmup_time_figure()
    #make_acvf_figures()