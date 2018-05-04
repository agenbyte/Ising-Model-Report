'''
Contains code for simulating the Ising model
on a square grid.
'''
import numpy as np


class Simulation(object):
    '''
    The Simulation object stores data about the running simulation.
    It also allows us to do some cool things.
    If you give it your own configuration, it must have periodic
    boundary conditions already
    '''
    def __init__(self, size, temperature, config=None):
        if config is None:
            self.config = self.generate_random_lattice(size)
        else:
            self.config = config
        self.size = size
        self.temperature = temperature
        self.energy_history = []
        self.energy = self.calculate_energy()
        self.energy_history.append(self.energy)

    def step(self):
        lattice_site = self.choose_random_lattice_site()
        energy_change = self.compute_energy_change(lattice_site)
        acceptance_probability = min(1, np.exp(self.temperature*energy_change))
        if np.random.uniform() < acceptance_probability:
            self.flip_lattice_site(lattice_site)
            self.energy += energy_change
            self.config = set_period_boundary(self.config)

    def step_n_times(self, nsteps):
        for __ in range(nsteps):
            self.step()
            self.energy_history.append(self.energy)

    def calculate_energy(self):
        '''
        Computes the energy of the system
        '''
        energy = 0
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                energy += self.config[i-1, j] * self.config[i, j] +\
                          self.config[i+1, j] * self.config[i, j] +\
                          self.config[i, j-1] * self.config[i, j] +\
                          self.config[i, j+1] * self.config[i, j]
        return energy/4.0

    def generate_random_lattice(self, size):
        '''
        Generates a 2-dimensional lattice of -1's and 1's
        and applies periodic boundary conditions to the system
        '''
        size = size + 2
        config = 2*np.random.randint(0, high=2, size=(size, size)) - 1
        config = set_period_boundary(config)
        return config

    def flip_lattice_site(self, lattice_site):
        self.config[lattice_site] *= -1

    def compute_energy_change(self, lattice_site):
        print(self.config[lattice_site])
        print(lattice_site)
        left = lattice_site - (1, 0)
        right = lattice_site + (1, 0)
        up = lattice_site + (0, 1)
        down = lattice_site - (0, 1)
        delta_energy = -2 * self.config[lattice_site] *\
                            (self.config[left] + self.config[right] +\
                             self.config[up] + self.config[down])
        print(delta_energy)
        return delta_energy

    def choose_random_lattice_site(self):
        flip_site = np.random.randint(1, high=self.size+1, size=(2))
        return flip_site



def set_period_boundary(config):
    config[:, -1] = config[:, 1]
    config[:, 0] = config[:, -2]
    config[-1, :] = config[1, :]
    config[0, :] = config[-2, :]
    return config
