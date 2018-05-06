import numpy as np
import matplotlib.pyplot as plt

from ising import Simulation


a = Simulation(16, 1000)
a.step_n_times(100000)

plt.plot(np.array(a.energy_history)/(a.config.shape[0]**2))
plt.show()
plt.pcolor(a.config)
plt.show()

