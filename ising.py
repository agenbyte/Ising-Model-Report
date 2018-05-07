import numpy as np
import matplotlib.pyplot as plt


SIZE = 2**4
NITERATIONS = 2**10*SIZE
NTEMPS = 2**8


TEMPERATURE_ARRAY = np.random.normal(2.26919, .64, NTEMPS)
ENERGY_ARRAY = np.zeros(NTEMPS)
MAGN_ARRAY = np.zeros(NTEMPS)
MAGS_ARRAY = np.zeros(NTEMPS)

for ii, T in enumerate(TEMPERATURE_ARRAY):
    lat = 2*np.random.randint(2, size=(SIZE+2, SIZE+2))-1
    ehist = np.zeros(NITERATIONS+1)
    mhist = np.zeros(NITERATIONS+1)


    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0

    energy = 0
    for i in range(1, SIZE+1):
        for j in range(1, SIZE+1):
            neighbors = np.array(lat[i,j]!=np.array(
                [lat[i+1, j], lat[i-1, j], lat[i,j+1], lat[i,j-1]]
            ), dtype=int)
            energy += sum(neighbors)
    energy /= 2.0

    ehist[0] = energy
    mhist[0] = abs(np.sum(lat))/SIZE**2

    for step in range(NITERATIONS):

        si, sj = np.random.randint(1, SIZE+1, size=2)
        neighbors = np.array(lat[si,sj]!=np.array(
                [lat[si+1, sj], lat[si-1, sj], lat[si,sj+1], lat[si,sj-1]]
            ), dtype=int)
        delta_E = 4 - 2 * sum(neighbors)
        p = min(1, np.exp(-delta_E/T))
        if np.random.uniform() < p:
            lat[si, sj] *= -1
            ehist[step+1] = ehist[step] + delta_E
            mhist[step+1] = abs(np.sum(lat[1:-1,1:-1])/SIZE**2)
            lat[ 0,  1:-1] = lat[-2,  1:-1]
            lat[-1,  1:-1] = lat[ 1,  1:-1]
            lat[ 1:-1,  0] = lat[ 1:-1, -2]
            lat[ 1:-1, -1] = lat[ 1:-1,  1]
            lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0
        else:
            ehist[step+1] = ehist[step]
            mhist[step+1] = abs(np.sum(lat[1:-1,1:-1])/SIZE**2)

    ehist /= SIZE**2
    # plt.plot(ehist)
    # plt.show()
    # plt.pcolor(lat)
    # plt.show()
    e = ehist[2**9:].mean()
    m = mhist[2**9:].mean()
    ENERGY_ARRAY[ii] = e
    MAGN_ARRAY[ii] = m

plt.scatter(TEMPERATURE_ARRAY, ENERGY_ARRAY)
plt.show()

