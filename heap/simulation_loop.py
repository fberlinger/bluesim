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
import os, glob

from predator import Predator
from environment import Environment
from dynamics import Dynamics
from lib_heap import Heap


def exp_rv(param):
    """Draws a uniform random number between 0 and 1 and returns an exponentially distributed random number with parameter param.

    Args:
        param (float): Parameter of exponentially distributed random number

    Returns:
        float: Exponentially distributed random number
    """
    x = random.random()
    return -math.log(1-x)/param

def log_meta():
    """Logs the meta data of the experiment
    """
    meta = {'Experiment': experiment_type, 'loopname':loopname, 'Number of fishes': no_fish, 'Simulation time [s]': simulation_time, 'Clock frequency [Hz]': clock_freq, 'Arena [mm]': arena_list, 'Visual range [mm]': v_range, 'Width of blindspot [mm]': w_blindspot, 'Radius of blocking sphere [mm]': r_sphere, 'Visual noise magnitude [% of distance]': n_magnitude, 'parsing' : parsing_bool, 'pred_bool': pred_bool, 'escape_angle':escape_angle, 'surface_reflections': surface_reflections, 'pred_speed': pred_speed}
    with open('./logfiles/{}_meta.txt'.format(filename), 'w') as f:
        json.dump(meta, f, indent=2)

# Read Experiment Description
try:
    experiment_type = sys.argv[1]
except:
    experiment_type = 'aligning'#'fountain' #
    print('No experiment description provided, using as default', experiment_type)

Fish = getattr(importlib.import_module('fishfood.' + experiment_type), 'Fish') #import Fish class directly from module specified by experiment type

#remove all previous kf log files
for kf_file in glob.glob("./logfiles/kf*"):
    os.remove(kf_file)

simulation_time = 60 # [s]
clock_freq = 0.75 # [Hz]
clock_rate = 1/clock_freq

# Fish Specifications
v_range=3000 # visual range, [mm]
w_blindspot=50 # width of blindspot, [mm]
r_sphere=50 # radius of blocking sphere for occlusion, [mm]
n_magnitude=0.05 # visual noise magnitude, [% of distance] #0.05
surface_reflections=True
parsing_bool = True

if experiment_type == "fountain":
    pred_bool = True
    escape_angle = 110 * math.pi/180 # escape angle for fish, [rad]
    fish_factor_speed = 0.05 #slow down fish from max speed with this factor, keep at 0.05
    pred_speed = 50 # [mm/s] (good range: 40-50, max fish speed is approx 60mm/s with fish_factor_speed = 0.05)
else:
    pred_bool = False
    escape_angle = []
    fish_factor_speed = []
    pred_speed = []

fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude, surface_reflections, parsing_bool, escape_angle, pred_speed, fish_factor_speed)

# Standard Tank
arena_list = [1780, 1780, 1170]
#arena_list = [2*x for x in arena_list] #pw for fountain more space for now
arena = np.array(arena_list)
arena_center = arena / 2.0
initial_spread = 500

# Experimental Parameters
loopname = 'aligning_clockfreq'
no_repetitions = 5
no_fish_range = [8]#[10,20]

seed_array = np.array([[ 2, 25, 15, 25, 12, 68, 26, 99, 97, 81, 24, 79, 18, 26, 69, 59,
        76, 38, 48, 52],
       [60, 92, 83, 90, 63, 20, 53, 42, 33, 52, 56, 70, 66, 84, 95, 27,
        57, 26,  4,  3],
       [ 0, 64, 72, 30, 70, 93, 44, 96, 92,  5,  0, 68,  2, 31, 22, 73,
        70,  7, 90, 94]]) #to ensure that all loops have same initial conditions, generated with  np.random.randint(100, size = [3,20])
i = 0

for no_fish in sorted(list(no_fish_range) * no_repetitions):

    # Standard Surface Initialization
    pos = np.zeros((no_fish, 4))
    vel = np.zeros((no_fish, 4))
    np.random.seed(seed_array[0, i])
    pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
    np.random.seed(seed_array[1, i])
    pos[:,2] = initial_spread * np.random.rand(1, no_fish) + 100# z, all fish a same noise-free depth results in LJ lock
    np.random.seed(seed_array[2, i])
    pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5)# phi
    i+=1
    # Create Environment, Dynamics, And Heap
    environment = Environment(pos, vel, fish_specs, arena, pred_bool, clock_freq)
    dynamics = Dynamics(environment)

    H = Heap(no_fish + pred_bool)

    # Create Fish Instances And Insert Into Heap
    fishes = []
    for fish_id in range(no_fish):
        clock = exp_rv(clock_rate)
        fishes.append(Fish(fish_id, dynamics, environment))
        H.insert(fish_id, clock)

    if pred_bool:
        clock = exp_rv(clock_rate)
        predator = Predator(dynamics, environment)
        H.insert(no_fish, clock) #insert one more ticket in heap
    # Simulate
    print('#### WELCOME TO BLUESIM ####')
    print('Progress:', end=' ', flush=True)
    t_start = time.time()

    simulation_steps = (no_fish+pred_bool)*simulation_time*clock_freq # overall

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
        if uuid < no_fish:
            fishes[uuid].run(duration)
        else:
            predator.run(duration)
        H.insert(uuid, event_time + duration)

        steps += 1

    print('| Duration: {} sec\n -'.format(round(time.time()-t_start)))


    # Save Data
    filename = time.strftime("%y%m%d_%H%M%S") # date_time
    environment.log_to_file(filename)
    log_meta()
    os.system('python3 statistical_analysis.py')

    print('Simulation data got saved in ./logfiles/{}_data.txt,\nand corresponding experimental info in ./logfiles/{}_meta.txt.\n -'.format(filename, filename))
    print('Create corresponding animation by running >python animation.py {}'.format(filename))
    print('#### GOODBYE AND SEE YOU SOON AGAIN ####')