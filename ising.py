import numpy as np
import matplotlib.pyplot as plt


SIZE = 32
NITERATIONS = 50 * SIZE**2
TEMPERATURE = 5.26919
NTEMPS = 11


TEMPERATURE_ARRAY = np.linspace(0.1, 2, NTEMPS)
ENERGY_ARRAY = np.zeros(NTEMPS)
MAGN_ARRAY = np.zeros(NTEMPS)
MAGS_ARRAY = np.zeros(NTEMPS)
BASE_LAT = 2*np.random.randint(2, size=(SIZE+2, SIZE+2))-1

for ii, T in enumerate(TEMPERATURE_ARRAY):
    # Set up a new lattice
    print(ii)
    lat = BASE_LAT
    ehist = np.zeros(NITERATIONS+1)
    mhist = np.zeros(NITERATIONS+1)
    mshist = np.zeros(NITERATIONS)
    shhist = np.zeros(NITERATIONS)

    # Now set the periodic boundary conditions
    lat[ 0,  1:-1] = lat[-2,  1:-1]
    lat[-1,  1:-1] = lat[ 1,  1:-1]
    lat[ 1:-1,  0] = lat[ 1:-1, -2]
    lat[ 1:-1, -1] = lat[ 1:-1,  1]
    lat[0, 0] = lat[-1, 0] = lat[0, -1] = lat[-1, -1] = 0

    energy = 0
    for i in range(1, SIZE+1):
        for j in range(1, SIZE+1):
            neighbors = lat[i+1, j] + lat[i-1, j] + lat[i,j+1] + lat[i,j-1]
            energy += lat[i, j] * neighbors
    energy /= -2.0

    ehist[0] = energy

    for step in range(NITERATIONS):
        si, sj = np.random.randint(1, SIZE+1, size=2)
        delta_E = 2 * lat[si, sj] * (lat[si+1,sj] + lat[si-1,sj] + lat[si,sj+1] + lat[si,sj-1])
        p = min(1, np.exp(-delta_E/T))
        if np.random.uniform() < p:
            lat[si, sj] *= -1
            ehist[step+1] = ehist[step] + delta_E
            mhist[step+1] = abs(np.sum(lat)/SIZE**2)
        else:
            ehist[step+1] = ehist[step] + delta_E
            mhist[step+1] = abs(np.sum(lat)/SIZE**2)
    plt.plot(ehist)
    plt.show()
    e = ehist[1000:].mean()
    m = mhist[1000:].mean()
    ENERGY_ARRAY[ii] = e
    MAGN_ARRAY[ii] = m

plt.plot(ENERGY_ARRAY)
plt.show()