# Main functions for Ising simulation

import numpy as np
import matplotlib.pyplot as plt


def ising_step(lat, size, temperature):
    '''
    Does one Metropolis step for the simulation.
    Returns the resulting lattice and the energy change.
    '''
    si, sj = np.random.randint(1, size+1, size=2)
    neighbors = np.array(lat[si,sj]!=np.array(
            [lat[si+1, sj], lat[si-1, sj], lat[si,sj+1], lat[si,sj-1]]
        ), dtype=int)
    delta_E = 4 - 2*sum(neighbors)
    p = min(1, np.exp(-delta_E/temperature))
    if np.random.uniform() < p:
        lat[si, sj] *= -1
        lat[ 0,  1:-1] = lat[-2,  1:-1]
        lat[-1,  1:-1] = lat[ 1,  1:-1]
        lat[ 1:-1,  0] = lat[ 1:-1, -2]
        lat[ 1:-1, -1] = lat[ 1:-1,  1]
        lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
    else:
        delta_E = 0
    return lat, delta_E


def ising_simulation(size, temperature, niterations, lat):
    '''
    Does N Metropolis steps of the ising model. Returns
    energy and magnetization histories.
    '''
    ehist = np.zeros(niterations+1)
    mhist = np.zeros(niterations+1)
    e0 = calc_energy(lat, size)
    ehist[0] = e0
    for step in range(niterations):
        lat, delta_E = ising_step(lat, size, temperature)
        ehist[step+1] = ehist[step] + delta_E
        mhist[step+1] = abs(np.sum(lat[1:-1,1:-1])/size**2)
    return ehist, mhist


def calc_energy(lat, size):
    '''
    Calculates the energy of the given lattice
    '''
    energy = 0
    for i in range(1, size+1):
        for j in range(1, size+1):
            neighbors = np.array(lat[i,j]!=np.array(
                [lat[i+1, j], lat[i-1, j], lat[i,j+1], lat[i,j-1]]
            ), dtype=int)
            energy += sum(neighbors)
    energy /= 2.0
    return energy


def compute_ACcF(data, start, end, steps):
    '''
    Computes the autocorrelation of the data.
    '''
    ACvF = []
    for i in range(steps):
        cov = np.corrcoef(data[start:end], data[start+i:end+i])[0][1]
        ACvF.append(cov)
    return ACvF