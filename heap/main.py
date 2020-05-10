import math
import random
import time
import numpy as np

from fish import Fish
from environment import Environment
from dynamics import Dynamics
from lib_heap import Heap

def exp_rv(param):
    """Draw a uniform random number between 0 and 1 and returns an exponentially distributed random number with parameter param.
    
    Args:
        param (float): Parameter of exponentially distributed random number
    
    Returns:
        float: Exponentially distributed random number
    """
    x = random.random()
    return -math.log(1-x)/param


# Experimental Parameters
no_fish = 20
simulation_time = 100 # [s]
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

# Create Environment, Dynamics, And Heap
environment = Environment(pos, vel, arena)
dynamics = Dynamics(environment, clock_freq)
H = Heap(no_fish)

# Create Fish Instances And Insert Into Heap
fishes = []
for fish_id in range(no_fish):
    clock = exp_rv(clock_rate)
    fishes.append(Fish(fish_id, dynamics, environment))
    H.insert(fish_id, clock)

# Simulate
print('#### WELCOME TO BLUESIM ####')
print('Progress:', end=' ', flush=True)
t_start = time.time()
simulation_steps = no_fish*simulation_time*clock_freq # overall
steps = 0
prog_incr = 0.1

while True:
    progress = steps/simulation_steps
    if progress >= prog_incr:
        print('{}%'.format(round(prog_incr*100)), end=' ', flush=True)
        prog_incr += 0.1
    if steps >= simulation_steps:
            break

    (uuid, event_time) = H.delete_min()
    fishes[uuid].run()
    next_clock = event_time + exp_rv(clock_rate)
    H.insert(uuid, next_clock)

    steps += 1

print('| Duration: {} sec\n -'.format(round(time.time()-t_start)))

# Save Data
filename = time.strftime("%y%m%d_%H%M%S") # date_time
environment.log_to_file(filename)

print('Simulation data got saved in ./logfiles/{}_data.txt\n -'.format(filename))
print('Create corresponding animation by running >python animation.py {}'.format(filename))
print('#### GOODBYE AND SEE YOU SOON AGAIN ####')