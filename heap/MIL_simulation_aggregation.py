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

from MIL_environment import Environment
from dynamics import Dynamics
from lib_heap import Heap
from circle_eval import *


def log_meta():
    """Logs the meta data of the experiment
    """
    meta = {'Experiment': experiment_file, 'Number of fishes': no_fishes, 'Simulation time [s]': simulation_time, 'Clock frequency [Hz]': clock_freq, 'Arena [mm]': arena_list, 'Visual range [mm]': v_range, 'Width of blindspot [mm]': w_blindspot, 'Radius of blocking sphere [mm]': r_sphere, 'Visual noise magnitude [% of distance]': n_magnitude}
    with open('./logfiles/{}_meta.txt'.format(filename), 'w') as f:
        json.dump(meta, f, indent=2)

def log_series(series):
    """Logs the meta data of the experiment
    """
    np.savetxt('./logfiles/{}_{}_series.txt'.format(len(clock_freqs), test_runs), series, fmt='%.2f', delimiter=',')

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
simulation_time = 600 # [s]
clock_freqs = [2] # [Hz]
no_fishes = 10
alpha = 0
test_runs = 1

# Fish Specifications
v_range = 100000 # visual range, [mm]
w_blindspot = 50 # width of blindspot, [mm]
r_sphere = 50 # radius of blocking sphere for occlusion, [mm]
n_magnitude = 0.1 # visual noise magnitude, [% of distance]
fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)

# Standard Tank
arena_list = [elem * 10 for elem in [1780, 1780, 1170]] # 2000
arena = np.array(arena_list)
arena_center = arena / 2.0

# STATISTICS
print('#### WELCOME TO BLUESIM ####')
for freq_iter, clock_freq in enumerate(clock_freqs):
    clock_rate = 1/clock_freq
    for test_run in range(test_runs):
        # Standard Surface Initialization
        initial_spread = 1000
        pos = np.zeros((no_fishes, 4))
        vel = np.zeros((no_fishes, 4))
        pos[:,:2] = initial_spread * (np.random.rand(no_fishes, 2) - 0.5) + arena_center[:2] # x,y
        pos[:,2] = 10 * np.random.rand(1, no_fishes) # z, all fish at same noise-free depth results in LJ lock
        pos[:,3] = 2*math.pi * (np.random.rand(1, no_fishes) - 0.5) # phi

        # Create Environment, Dynamics, And Heap
        environment = Environment(pos, vel, fish_specs, arena)
        dynamics = Dynamics(environment)
        H = Heap(no_fishes)

        # Create Fish Instances And Insert Into Heap
        fishes = []
        for fish_id in range(no_fishes):
            clock = random.gauss(clock_rate, 0.1*clock_rate)
            fishes.append(Fish(fish_id, dynamics, environment))
            H.insert(fish_id, clock)

        # Simulate
        print('Simulation {} out of {}: clock_freq = {}, test_run = {}.'.format(freq_iter*test_runs+test_run+1, len(clock_freqs)*test_runs, clock_freq, test_run+1))
        print('Progress:', end=' ', flush=True)
        t_start = time.time()
        simulation_steps = no_fishes*simulation_time*clock_freq # overall
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
        filename = str(clock_freq) + '_' + str(test_run)
        environment.log_to_file(filename)
        #x_c, y_c, r_c, error, max_r, min_r, t_convergence = calc_circle(environment.tracking[:, :no_fishes*4], clock_freq, test_run)

        #if freq_iter*test_runs+test_run+1 == 1:
        #    series = np.array([x_c[0], y_c[0], r_c[0], error, max_r[0], min_r[0], t_convergence]).reshape(1,7)
        #else:
        #    series = np.concatenate((series, np.array([x_c[0], y_c[0], r_c[0], error, max_r[0], min_r[0], t_convergence]).reshape(1,7)), axis=0)
        log_meta()

#log_series(series)

print(' -\nSimulation data got saved in ./logfiles/*_data.txt,\nand corresponding experimental info in ./logfiles/*_meta.txt.\n -')
print('Create corresponding animation by running >python3 MIL_animation.py filename')
print('#### GOODBYE AND SEE YOU SOON AGAIN ####')
