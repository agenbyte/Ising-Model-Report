import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


class Simulation(object):

    # This function is called when you create a Simulation object
    # a = Simulation(100, 100)
    # will create a new simulation with size 100 and temperature
    # 100. We dont config has a default of None which I set to
    # generate a random spin configuration.
    def __init__(self, size, temperature, config=None, periodic=True):
        if config is None:
            self.config = self.generate_random_lattice(size, periodic=True)
        else:
            self.config = config
        self.size = size
        # So this isn't really temperature anymore but like whatevers
        # TODO
        self.periodic = periodic
        self.temperature = temperature
        self.energy_history = []
        self.energy = self.calculate_energy()
        self.energy_history.append(self.energy)

    def calculate_energy(self):
        '''
        Computes the energy of the system. Avoiding using this
        method frequently. It is very slow.
        '''
        # There are two things we need to worry about here. We need to
        # know if there are periodic boundary conditions going on and
        # thats it just one thing
        if not self.periodic:
            # Do something different. IDK yet
            pass
        else:
            energy = 0
            for i in range(1, self.size-1):
                for j in range(1, self.size-1):
                    energy += self.config[i-1, j] * self.config[i, j] +\
                              self.config[i+1, j] * self.config[i, j] +\
                              self.config[i, j-1] * self.config[i, j] +\
                              self.config[i, j+1] * self.config[i, j]
            return energy/4.0

    def simulate_me_daddy(self, nsteps):
        for __ in range(nsteps):
            self.step()
            self.energy_history.append(self.energy)

    def generate_random_lattice(self, size, periodic=False):
        '''
        Generates a 2-dimensional lattice of -1's and 1's
        and applies periodic boundary conditions to the system
        '''
        if periodic:
            size = size + 2
        config = 2*np.random.randint(0, high=2, size=(size, size)) - 1
        if periodic:
            config = self._set_ghost_cells(config)
        return config

    def step(self):
        # Need to enforce the boundary conditions
        # Choose a site
        lattice_site = self.choose_random_lattice_site()
        # Now we need to compute the change in energy if we were to flip
        # that lattice site
        delta_energy = self.get_energy_change_from_lattice_flip(lattice_site)
        acceptance_probability = min(1, np.exp(self.temperature*delta_energy))
        if np.random.uniform() < acceptance_probability:
            self.flip_lattice_site(lattice_site)
            self.energy += delta_energy
            self.config = self._set_ghost_cells(self.config)

    def flip_lattice_site(self, lattice_site):
        self.config[lattice_site] *= -1


    def get_energy_change_from_lattice_flip(self, lattice_site):
        x = lattice_site[0]
        y = lattice_site[1]
        delta_energy = 2 * (-self.config[x, y] * self.config[x-1, y] +\
                            -self.config[x, y] * self.config[x+1, y] +\
                            -self.config[x, y] * self.config[x, y-1] +\
                            -self.config[x, y] * self.config[x, y+1])
        return delta_energy

    def choose_random_lattice_site(self):
        fi, fj = np.random.randint(1, high=self.size+1, size=(2))
        return fi, fj

    def _set_ghost_cells(self, config):
        left_slice = -1
        right_slice = 1
        config[:, left_slice] = config[:, right_slice]
        config[:, 0] = config[:, -2]
        config[-1, :] = config[:, 1]
        config[0, :] = config[-2, 0]
        return config
