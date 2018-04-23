import numpy as np


def do_the_ising_model(temperature, num_steps, size=100):
    config = np.zeros((size+2, size+2))
    config[1:-1, 1:-1] = 2*np.random.randint(0, high=2, size=(size, size)) - 1
    # Add the ghost cells
    # The corners are incorrect but that doesn't matter really
    config[:,-1] = config[:,1]
    config[:,0] = config[:,-2]
    config[-1,:] = config[:,1]
    config[0,:] = config[:,-2]

    # Heres the main loop
    energy_history = []
    for __ in range(num_steps):
        energy = 0
        for i in range(1, size+1):
            for j in range(1, size+1):
                energy += config[i-1, j] * config[i, j] +\
                          config[i+1, j] * config[i, j] +\
                          config[i, j-1] * config[i, j] +\
                          config[i, j+1] * config[i, j]
        energy /= 4.0
        energy_history.append(energy)
        # Now pick a random site to flip
        fi, fj = np.random.randint(1, high=size+2, size=(2))
        config[fi, fj] *= -1
        new_energy = 0
        for i in range(1, size+1):
            for j in range(1, size+1):
                new_energy += config[i-1, j] * config[i, j] +\
                              config[i+1, j] * config[i, j] +\
                              config[i, j-1] * config[i, j] +\
                              config[i, j+1] * config[i, j]
        new_energy /= 4.0

        if new_energy > energy:
            accept = np.random.uniform(0, 1)
            if accept < np.exp(temperature*(energy-new_energy)):
                pass
            else:
                config[fi, fj] *= -1
    return energy_history