import numpy as np
import matplotlib.pyplot as plt


SIZE = 2**4
NITERATIONS = 2**10*SIZE**2
NTEMPS = 2**4


TEMPERATURE_ARRAY = np.random.normal(1.13459265711, .64, NTEMPS)
ENERGY_ARRAY = np.zeros(NTEMPS)
MAGN_ARRAY = np.zeros(NTEMPS)
MAGS_ARRAY = np.zeros(NTEMPS)

def ising_simulation(size, temperature, niterations, lat):
    ehist = np.zeros(niterations+1)
    mhist = np.zeros(niterations+1)
    e0 = calc_energy(lat, size)
    ehist[0] = e0
    for step in range(niterations):
        si, sj = np.random.randint(1, size+1, size=2)
        neighbors = np.array(lat[si,sj]!=np.array(
                [lat[si+1, sj], lat[si-1, sj], lat[si,sj+1], lat[si,sj-1]]
            ), dtype=int)
        delta_E = 4 - 2*sum(neighbors)
        p = min(1, np.exp(-delta_E/temperature))
        if np.random.uniform() < p:
            lat[si, sj] *= -1
            ehist[step+1] = ehist[step] + delta_E
            mhist[step+1] = abs(np.sum(lat[1:-1,1:-1])/size**2)
            # Reset the periodic boundary conditions
            lat[ 0,  1:-1] = lat[-2,  1:-1]
            lat[-1,  1:-1] = lat[ 1,  1:-1]
            lat[ 1:-1,  0] = lat[ 1:-1, -2]
            lat[ 1:-1, -1] = lat[ 1:-1,  1]
            lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
        else:
            ehist[step+1] = ehist[step]
            mhist[step+1] = abs(np.sum(lat[1:-1,1:-1])/size**2)
    return ehist, mhist

def calc_energy(lat, size):
    energy = 0
    for i in range(1, size+1):
        for j in range(1, size+1):
            neighbors = np.array(lat[i,j]!=np.array(
                [lat[i+1, j], lat[i-1, j], lat[i,j+1], lat[i,j-1]]
            ), dtype=int)
            energy += sum(neighbors)
    energy /= 2.0
    return energy


if __name__ == '__main__':
    for ii, T in enumerate(TEMPERATURE_ARRAY):
        lat = 2*np.random.randint(2, size=(SIZE+2, SIZE+2))-1
        ehist = np.zeros(NITERATIONS+1)
        mhist = np.zeros(NITERATIONS+1)


        lat[ 0,  1:-1] = lat[-2,  1:-1]
        lat[-1,  1:-1] = lat[ 1,  1:-1]
        lat[ 1:-1,  0] = lat[ 1:-1, -2]
        lat[ 1:-1, -1] = lat[ 1:-1,  1]
        lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0

        energy = calc_energy(lat, SIZE)

        ehist[0] = energy
        mhist[0] = abs(np.sum(lat))/SIZE**2

        ehist, mhist = ising_simulation(SIZE, T, NITERATIONS, lat)
        plt.plot(ehist)
        plt.show()

        ehist /= SIZE**2
        e = ehist[2**9:].mean()
        m = mhist[2**9:].mean()
        ENERGY_ARRAY[ii] = e
        MAGN_ARRAY[ii] = m

    plt.scatter(TEMPERATURE_ARRAY, ENERGY_ARRAY)
    plt.show()



def compute_ACcF(data, start, end, steps):
    ACvF = []
    for i in range(steps):
        cov = np.corrcoef(data[start:end], data[start+i:end+i])[0][1]
        ACvF.append(cov)
    return ACvF