import os

import numpy as np
import matplotlib.pyplot as plt

import ising_sw


data_directory = 'Data'
# Lets do a smaller simulation first. Otherwise this is going to take for fucking ever
NSIMS = 100
SIZE = 16
TEMP =2.26919 # This is the critical temperature
# Do 32 full sweeps of the lattice to get good data
NSTEP = 2 ** 5 * SIZE**2


en_arr = np.zeros(shape=(NSIMS, NSTEP+1))
print("STARTING THE SIMULATION")
for i in range(NSIMS):
    LAT = 2 * np.random.randint(0, 2, size=(SIZE+2, SIZE+2)) - 1
    en_arr[i, :], __ = ising_sw.ising_simulation(SIZE, TEMP, NSTEP, LAT)
    print("FINISHED STEP {}".format(i))

np.savez(os.path.join(data_directory, 'acvf_averaging_data.npz'), en_arr)
