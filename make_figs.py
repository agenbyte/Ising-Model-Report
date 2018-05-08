import os

import numpy as np
import matplotlib.pyplot as plt

import ising

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
    e, __ = ising.ising_simulation(size, temperature, nsteps, lat)
    e /= size**2
    plt.figure(figsize=(5,4))
    plt.plot(e)
    plt.xlabel('Time-Step')
    plt.ylabel('Energy per Atom')
    plt.show()
    #plt.savefig(out_path)


def make_pcolor_figures():
    size = 100
    tc = 1.1346
    t0 = 0.25 * tc
    t1 = tc
    t2 = 2 * tc
    base_lattice = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
    base_lattice[ 0,  1:-1] = base_lattice[-2,  1:-1]
    base_lattice[-1,  1:-1] = base_lattice[ 1,  1:-1]
    base_lattice[ 1:-1,  0] = base_lattice[ 1:-1, -2]
    base_lattice[ 1:-1, -1] = base_lattice[ 1:-1,  1]
    base_lattice[0, 0] = base_lattice[-1, 0] = base_lattice[0, -1] = base_lattice[-1, -1] = 0

    lat = np.copy(base_lattice)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
    ax1.pcolor(lat[1:-1, 1:-1])
    ax1.set_title('t=1')
    ax2.set_title('t=10000')
    ax3.set_title('t=100000')
    ax1.set_ylabel(r'$T=0.25T_c$')
    ax4.set_ylabel(r'$T=T_c$')
    ax7.set_ylabel(r'$T=2T_c$')

    for __ in range(10000):
        print('simming')
        lat, __ = ising.ising_step(lat, size, t0)
    ax2.pcolor(lat[1:-1, 1:-1])

    for __ in range(90000):
        print('simming')
        lat, __ = ising.ising_step(lat, size, tc)
    ax3.pcolor(lat[1:-1, 1:-1])

    lat = np.copy(base_lattice)
    ax4.pcolor(lat[1:-1, 1:-1])

    for __ in range(10000):
        lat, __ = ising.ising_step(lat, size, t1)
    ax5.pcolor(lat[1:-1, 1:-1])

    for __ in range(90000):
        lat, __ = ising.ising_step(lat, size, t1)
    ax6.pcolor(lat[1:-1, 1:-1])

    lat = np.copy(base_lattice)
    ax7.pcolor(lat[1:-1, 1:-1])

    for __ in range(10000):
        lat, __ = ising.ising_step(lat, size, t2)
    ax8.pcolor(lat[1:-1, 1:-1])

    for __ in range(90000):
        lat, __ = ising.ising_step(lat, size, t2)
    ax9.pcolor(lat[1:-1, 1:-1])

    plt.show()


def make_pcolor_long_fig():
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    size = 100
    tc = 1.1346
    t0 = 0.25 * tc
    base_lattice = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
    base_lattice[ 0,  1:-1] = base_lattice[-2,  1:-1]
    base_lattice[-1,  1:-1] = base_lattice[ 1,  1:-1]
    base_lattice[ 1:-1,  0] = base_lattice[ 1:-1, -2]
    base_lattice[ 1:-1, -1] = base_lattice[ 1:-1,  1]
    base_lattice[0, 0] = base_lattice[-1, 0] = base_lattice[0, -1] = base_lattice[-1, -1] = 0
    ax1.set_title("t=1")
    ax2.set_title("t=10000")
    ax3.set_title("t=100000")
    ax4.set_title("t=1000000")
    plt.suptitle('Time steps of MCMC simulation ($T=0.25T_c$)')
    lat = np.copy(base_lattice)
    ax1.pcolor(lat[1:-1,1:-1])
    for __ in range(10000):
        lat, __ = ising.ising_step(lat, size, t0)
    ax2.pcolor(lat[1:-1,1:-1])
    for __ in range(90000):
        lat, __ = ising.ising_step(lat, size, t0)
    ax3.pcolor(lat[1:-1,1:-1])
    for __ in range(900000):
        lat, __ = ising.ising_step(lat, size, t0)
    ax4.pcolor(lat[1:-1,1:-1])
    plt.show()



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
        en_arr[i, :], __ = ising.ising_simulation(size, tc, nsteps, lat)
        print('FINISHED STEP {}'.format(i))

    acvf = []
    for i in range(nsims):
        print('CALC #{}'.format(i))
        acvf.append(ising.compute_ACcF(en_arr[i, :], 10000, 15000, 5000))

    acvf = np.array(acvf)
    av = []
    for i in range(acvf.shape[1]):

        av.append(acvf[:, i].mean())
    plt.figure(figsize=(4,4))
    plt.plot(av)
    plt.xlabel("Simulation Time Shift")
    plt.ylabel("Autocorrelation")
    plt.savefig('Figures/acvf.eps')


def make_energy_magnet_figures():
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Energy vs Temperature')
    ax2.set_title('Magnetization vs Temperature')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Energy')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Magnetization')
    size = 16
    niterations = 1000 * size**2 # Sweep over everything 2000 times
    warmup = 100 # Ignore the first 100 sweeps
    temp_crit = 1.1346
    ntemps = 256
    sample_temperatures = np.random.normal(temp_crit, 0.64, size=ntemps)
    sample_temperatures = sample_temperatures[np.where(sample_temperatures>0)]
    ntemps = len(sample_temperatures)
    ener_array = np.zeros(ntemps)
    magn_array = np.zeros(ntemps)

    for ii, temp in enumerate(sample_temperatures):
        print('Temperature {} of {}'.format(ii, ntemps))
        lat = 2 * np.random.randint(0, 2, size=(size+2, size+2)) - 1
        ehist = np.zeros(niterations+1)
        mhist = np.zeros(niterations+1)
        lat[ 0,  1:-1] = lat[-2,  1:-1]
        lat[-1,  1:-1] = lat[ 1,  1:-1]
        lat[ 1:-1,  0] = lat[ 1:-1, -2]
        lat[ 1:-1, -1] = lat[ 1:-1,  1]
        lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0

        ehist, mhist = ising.ising_simulation(size, temp, niterations, lat)
        ehist /= size ** 2
        ener_array[ii]=(ehist[warmup:].mean())
        magn_array[ii]=(mhist[warmup:].mean())

    ax1.scatter(sample_temperatures, ener_array)
    ax2.scatter(sample_temperatures, magn_array)
    plt.show()






if __name__ == '__main__':
    #make_pcolor_figures()
    #make_warmup_time_figure()
    #make_acvf_figures()
    #make_energy_magnet_figures()
    make_pcolor_long_fig()
