import numpy as np


class IsingSimulation(object):
    def __init__(self, N):
        self.state = 2*np.random.randint(0, 1, (N, N))-1
    
    def get_energy(self):
        pass
