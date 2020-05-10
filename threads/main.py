import math
import random
import time
import numpy as np

import threading
lock = threading.Lock()

from fish import Fish
from environment import Environment
from dynamics import Dynamics


# Experimental Parameters
no_fish = 50
simulation_time = 1000 # [s]
clock_freq = 2
clock_rate = 1/clock_freq

# Standard Tank
arena = np.array([1780, 1780, 1170])
arena_center = arena / 2.0

# Standard Surface Initialization
initial_spread = 500
pos = np.zeros((no_fish, 4))
vel = np.zeros((no_fish, 4))
pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish a same noise-free depth results in LJ lock
pos[:,3] = math.pi * np.random.rand(1, no_fish) # phi

# Create Environment And Dynamics
environment = Environment(pos, vel, arena)
dynamics = Dynamics(environment, clock_freq)


# Simulate
print('#### WELCOME TO BLUESIM ####')
t_start = time.time()
simulation_steps = simulation_time*clock_freq # per fish

fishes = []
for no in range(no_fish):
    fishes.append(Fish(no, simulation_steps, dynamics, environment))

threads = [] 
for fish in fishes:
    threads.append(threading.Thread(target=fish.start, args=(lock,)))
    threads[-1].start()

for thread in threads:
    thread.join()

print('Duration: {} sec\n -'.format(round(time.time()-t_start)))

# Save Data
filename = time.strftime("%y%m%d_%H%M%S") # date_time
environment.log_to_file(filename)

print('Simulation data got saved in ./logfiles/{}_data.txt\n -'.format(filename))
print('Create corresponding animation by running >python animation.py {}'.format(filename))
print('#### GOODBYE AND SEE YOU SOON AGAIN ####')