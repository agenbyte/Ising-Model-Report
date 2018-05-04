from ising import Simulation

temps = range(11)

e = []
for t in temps:
    print(t)
    a = Simulation(32, t)
    a.step_n_times(1000000)
    e.append(a.get_mean())

print(e)