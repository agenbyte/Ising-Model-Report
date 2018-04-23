import numpy as np
import matplotlib.pyplot as plt


def f(temperature, num_steps, size=100):
    config = np.zeros((size+2, size+2))
    print(config.shape)
    config[1:-1, 1:-1] = 2*np.random.randint(0, high=2, size=(size, size)) - 1
    plt.pcolor(config)
    plt.show()
    # Add the ghost cells
    # The corners are incorrect but that doesn't matter really
    config[:,-1] = config[:,1]
    config[:,0] = config[:,-2]
    config[-1,:] = config[:,1]
    config[0,:] = config[:,-2]

    # Heres the main loop
    energy_history = []
    initial_energy = compute_energy(config)
    # Then all we do is add the the energy difference if accepted
    energy = initial_energy
    for __ in range(num_steps):
        energy_history.append(energy)
        fi, fj = np.random.randint(1, high=size+1, size=(2))
        config[fi, fj] *= -1
        # need to make sure the ghost cells update correctly as well
        config[:,-1] = config[:,1]
        config[:,0] = config[:,-2]
        config[-1,:] = config[:,1]
        config[0,:] = config[:,-2]
        #new_energy = compute_energy(config)
        # it turns out, we can get the energy difference quite easily
        # since computing the energy takes a long time, we wont do that lol
        delta_E = 2 * 1 * (config[fi, fj] * config[fi-1, fj] +\
                           config[fi, fj] * config[fi+1, fj] +\
                           config[fi, fj] * config[fi, fj-1] +\
                           config[fi, fj] * config[fi, fj+1])
        accept = np.random.uniform() < (min(1, np.exp(temperature*delta_E)))
        if not accept:
            # flip it back
            config[fi, fj] *= -1
        else:
            energy += delta_E
    plt.pcolor(config)
    plt.show()
    return energy_history



def compute_energy(spin_configuration):
    size = spin_configuration.shape[0]
    energy = 0
    for i in range(1, size-1):
        for j in range(1, size-1):
            energy += spin_configuration[i-1, j] * spin_configuration[i, j] +\
                      spin_configuration[i+1, j] * spin_configuration[i, j] +\
                      spin_configuration[i, j-1] * spin_configuration[i, j] +\
                      spin_configuration[i, j+1] * spin_configuration[i, j]
    return energy/4.0