#!/usr/bin/python
"""Statistical analyses simulation data from logfiles with ipyvolume

Attributes:
    clock_freq (float): Clock frequency
    clock_rate (float): Clock rate
    colors (np-array of floats): Colors fish depending on their location
    fig (figure object): ipv figure
    fishes (int): Number of simulated fishes
    phi (float): Orientation angles
    timesteps (TYPE): Description
    x (float): x-positions
    y (float): y-positions
    z (float): z-positions
"""
import json
import os
import sys
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def log_stat(loopname, filename, fishes, escape_angle, surface_reflections, speed_ratio, phi_std_init, phi_std_end, eaten, no_tracks_avg, hull_area_max):
    """Logs the meta data of the experiment
    """
    with open('./logfiles/{}_stat.csv'.format(loopname), 'a+') as f:
        f.write(
            '{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format( #add: escape_angle, reflection on/off, speed ratio pred/fish, tracking error: how to measure?? dist track - closest groundtruth?
                filename,
                fishes,
                escape_angle,
                speed_ratio,
                phi_std_init,
                phi_std_end,
                eaten,
                no_tracks_avg,
                hull_area_max
            )
        )


# Load Data
try:
    filename = sys.argv[1]
except:
    list_dir = glob.glob('logfiles/*.txt')
    filename = sorted(list_dir)[-1][9:-9]
    print('filename not specified! automatically choosing newest file:', filename)
try:
    data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
    with open('./logfiles/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
except:
    print('Data file with prefix {} does not exist.\nProvide prefix of data you want to plot in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    sys.exit()


# Read Experimental Parameters
clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = meta['Number of fishes']
if 'loopname' in meta:
    loopname = meta['loopname']
else:
    loopname = filename#'no_loop'

escape_angle = meta['escape_angle']
surface_reflections = meta['surface_reflections']
pred_speed = meta['pred_speed']

#phi std, mean
phi_mean_cos = np.zeros((timesteps))
phi_mean_sin = np.zeros((timesteps))

for ii in range(fishes):
    phi_mean_cos += np.cos(data[:, 4*ii + 3])
    phi_mean_sin += np.sin(data[:, 4*ii + 3])

phi_mean_cos = phi_mean_cos/fishes
phi_mean_sin = phi_mean_sin/fishes
phi_mean = np.arctan2(phi_mean_sin, phi_mean_cos)
phi_std = np.sqrt(-np.log(phi_mean_sin**2 + phi_mean_cos**2))

print('The initial std phi is {0:.1f}rad.'.format(phi_std[0]))
print('The final std phi is {0:.1f}rad.'.format(phi_std[-1]))
print('The difference of mean phi is {0:.1f}rad.'.format(phi_mean[-1] - phi_mean[0]))

#check eating area: ellipse
pred_pos = data[:, 8*fishes : 8*fishes + 4]
eaten = []

for ii in range(fishes):
    a = 60 #semi-major axis in x
    b = 50 #semi-minor axis in y
    rel_pos = data[:, 4*ii :  4*ii + 3] - pred_pos[:, 0 : 3]
    rel_pos_rot = np.empty((timesteps, 2))
    rel_pos_rot[:, 0] = np.cos(pred_pos[:, 3])*rel_pos[:, 0] - np.sin(pred_pos[:, 3])*rel_pos[:, 1]
    rel_pos_rot[:, 1] = np.sin(pred_pos[:, 3])*rel_pos[:, 0] + np.cos(pred_pos[:, 3])*rel_pos[:, 1]
    p = ((rel_pos[:, 0])**2 / a**2 +  (rel_pos[:, 1])**2 / b**2)
    eaten.append(np.any(p < 1)) #and consider delta z !

print('{} fish out of {} got eaten.'.format(sum(eaten), fishes))

#log kf data¨
no_tracks_avg = 0
for protagonist_id in range(fishes):
    data_kf = np.genfromtxt('./logfiles/kf_{}.csv'.format(protagonist_id), delimiter=',')
    data_kf = data_kf[1:,:] #cut title row
    tracks = np.unique(data_kf[:,0]).astype(int)
    no_tracks = len(tracks) - 1 # -1 for pred
    no_tracks_avg += no_tracks

no_tracks_avg /= fishes
print('{} kf tracks were created in avg.'.format(no_tracks_avg))

#check pred_speed
v_max_avg = 0
for ii in range(fishes):
    v_xy = np.sqrt(data[:, 4*(ii+fishes)]**2 + data[:, 4*(ii+fishes) + 1]**2)
    v_max = max(v_xy)
    v_max_avg += v_max

v_max_avg /= fishes
speed_ratio = pred_speed/v_max_avg
print('The avg pred to fish speed ratio is {}.'.format(speed_ratio))

#calc ConvexHull
hull_area_max = 0
for i in range(timesteps):
    points = np.array([data[i, 4*ii :  4*ii + 2] for ii in range(fishes)])  # no_fish x 2
    hull = ConvexHull(points)
    if hull.volume > hull_area_max:
        hull_area_max = hull.volume

print('The largest hull area is {} m^2.'.format(hull_area_max/1000**2))

log_stat(loopname, filename, fishes, escape_angle, surface_reflections, speed_ratio, phi_std[0], phi_std[-1], int(sum(eaten)), no_tracks_avg, hull_area_max/1000**2)
