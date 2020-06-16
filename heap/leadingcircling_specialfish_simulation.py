"""Runs experiments. Change parameters here.

Attributes:
    arena (np-array of ints): Arena size in mm, [l, w, h]
    arena_center (np-array of floats): Arena center
    clock_freq (int): Fish update frequency
    clock_rate (float): Fish update rate
    dynamics (class instance): Fish dynamics
    environment (class instance): Fish environment
    filename (string): Time-prefix of logfiles, yymmdd_hhmmss
    fishes (list of class instances): Fishes
    H (class instance): Min heap
    initial_spread (int): Area spread of fish at initialization
    no_fish (int): Number of simulated fishes
    pos (np-array of floats): Fish positions at initialization, no_fish x [x,y,z,phi]
    prog_incr (float): Description
    simulation_steps (float): Required number of steps to simulate no_fish with clock_freq for simulation_time
    simulation_time (int): Experiment time in s
    steps (int): Simulation steps counter
    t_start (float): Experiment start time
    vel (np-array of floats): Fish velocities at initialization, no_fish x [vx,vy,vz,vphi]
"""
import json
import math
import numpy as np
import random
import sys
import time
import importlib

from environment import Environment
from dynamics import Dynamics
from lib_heap import Heap


def log_meta():
    """Logs the meta data of the experiment
    """
    meta = {'Experiment': experiment_file, 'Number of fishes': no_fish, 'Simulation time [s]': simulation_time, 'Clock frequency [Hz]': clock_freq, 'Arena [mm]': arena_list, 'Visual range [mm]': v_range, 'Width of blindspot [mm]': w_blindspot, 'Radius of blocking sphere [mm]': r_sphere, 'Visual noise magnitude [% of distance]': n_magnitude}
    with open('./logfiles/{}_meta.txt'.format(filename), 'w') as f:
        json.dump(meta, f, indent=2)

# Read Experiment Description
try:
    experiment_file = sys.argv[1]
except:
    print('Please provide the filename of the experiment you want to simulate, e.g.:\n >python simulation.py dispersion')
    sys.exit()

#import Fish class directly from module specified by experiment type
Fish = getattr(importlib.import_module('fishfood.' + experiment_file), 'Fish') 

## Feel free to loop over multiple simulations with different parameters! ##

# Experimental Parameters
no_fish = 50
simulation_time = 300 # [s]
clock_freq = 2 # [Hz]
clock_rate = 1/clock_freq

# Fish Specifications
v_range=3000 # visual range, [mm]
w_blindspot=50 # width of blindspot, [mm]
r_sphere=50 # radius of blocking sphere for occlusion, [mm]
n_magnitude=0.1 # visual noise magnitude, [% of distance]
fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)

# Standard Tank
arena_list = [17800, 17800, 11700]
arena = np.array(arena_list)
arena_center = arena / 2.0

# Standard Surface Initialization
initial_spread = 500
pos = np.zeros((no_fish, 4))
vel = np.zeros((no_fish, 4))
pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi

# Create Environment, Dynamics, And Heap
environment = Environment(pos, vel, fish_specs, arena)
dynamics = Dynamics(environment)
H = Heap(no_fish)

# Create Fish Instances And Insert Into Heap
fishes = []
for fish_id in range(no_fish):
    clock = random.gauss(clock_rate, 0.1*clock_rate)
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
    duration = random.gauss(clock_rate, 0.1*clock_rate)
    fishes[uuid].run(duration)
    H.insert(uuid, event_time + duration)

    steps += 1

print('| Duration: {} sec\n -'.format(round(time.time()-t_start)))

# Save Data
filename = time.strftime("%y%m%d_%H%M%S") # date_time
environment.log_to_file(filename)
log_meta()

print('Simulation data got saved in ./logfiles/{}_data.txt,\nand corresponding experimental info in ./logfiles/{}_meta.txt.\n -'.format(filename, filename))
print('Create corresponding animation by running >python animation.py {}'.format(filename))
print('#### GOODBYE AND SEE YOU SOON AGAIN ####')
