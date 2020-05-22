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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def init_log_stat():
    print('creating stat logfile')
    with open('./logfiles/{}_stat.csv'.format(loopname), 'w') as f:
        f.truncate()
        f.write('experiment, filename, runtime [s], no_fish, n_magnitude, surface_reflections, escape_angle [rad], pred_speed_ratio, phi_std_init, phi_std_end, hull_area_max [m^2], pred_eaten, #tracks/timestep avg, #tracks overall avg, kf pos tracking error avg [m], kf phi tracking error avg [rad]  \n')


def log_stat(experiment_type, loopname, filename, runtime, fishes, noise, surface_reflections, escape_angle, speed_ratio, phi_std_init, phi_std_end, hull_area_max, eaten, no_tracks_avg, no_new_tracks_avg, pos_tracking_error_avg, phi_tracking_error_avg):
    """Logs the meta data of the experiment
    """
    with open('./logfiles/{}_stat.csv'.format(loopname), 'a+') as f:
        f.write(
            '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format( 
                experiment_type,
                filename,
                runtime,
                fishes,
                noise,
                surface_reflections,
                escape_angle,
                speed_ratio,
                phi_std_init,
                phi_std_end,
                hull_area_max,
                eaten,
                no_tracks_avg,
                no_new_tracks_avg,
                pos_tracking_error_avg,
                phi_tracking_error_avg
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
n_magnitude = meta['Visual noise magnitude [% of distance]']
fishes = meta['Number of fishes']
escape_angle = meta['escape_angle']
surface_reflections = meta['surface_reflections']
pred_speed = meta['pred_speed']
pred_bool = meta['pred_bool']
experiment_type = meta['Experiment']


if 'loopname' in meta:
    loopname = meta['loopname']
else:
    loopname = filename#'no_loop'

#start logging
if not os.path.isfile('./logfiles/{}_stat.csv'.format(loopname)):
    init_log_stat()

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

print('The initial std phi is {0:.1f}.'.format(phi_std[0]))
print('The final std phi is {0:.1f}.'.format(phi_std[-1]))
print('The difference of mean phi is {0:.1f} deg.'.format(abs(phi_mean[-1] - phi_mean[0])*180/math.pi))

#check eating area: ellipse
eaten = []
speed_ratio = []
if pred_bool:
    pred_pos = data[:, 8*fishes : 8*fishes + 4]
    eaten = 0
    for ii in range(fishes):
        a = 60 #semi-major axis in x
        b = 50 #semi-minor axis in y
        rel_pos = data[:, 4*ii :  4*ii + 3] - pred_pos[:, 0 : 3]
        rel_pos_rot = np.empty((timesteps, 2))
        rel_pos_rot[:, 0] = np.cos(pred_pos[:, 3])*rel_pos[:, 0] - np.sin(pred_pos[:, 3])*rel_pos[:, 1]
        rel_pos_rot[:, 1] = np.sin(pred_pos[:, 3])*rel_pos[:, 0] + np.cos(pred_pos[:, 3])*rel_pos[:, 1]
        p = ((rel_pos[:, 0])**2 / a**2 +  (rel_pos[:, 1])**2 / b**2)
        eaten += np.any(p < 1) #and consider delta z !
    print('{} fish out of {} got eaten.'.format(eaten, fishes))
    
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

print('The largest hull area is {:0.5f} m^2.'.format(hull_area_max/1000**2))

#log kf data¨
no_tracks_avg = 0
no_new_tracks_avg = 0
pos_tracking_error_avg = 0 
phi_tracking_error_avg = 0   
  
for protagonist_id in range(fishes):
    data_kf = np.genfromtxt('./logfiles/kf_{}.csv'.format(protagonist_id), delimiter=',')
    data_kf = data_kf[1:,:] #cut title row
    #no tracks
    tracks = np.unique(data_kf[:,0]).astype(int)
    no_new_tracks = len(tracks) - pred_bool
    no_new_tracks_avg += no_new_tracks
    #tracking   
    kf_iterations = np.unique(data_kf[:,1]).astype(int)
    for i in range(timesteps):#kf_iterations: #all iterations
        kf = data_kf[np.argwhere(data_kf[:,1] == i).ravel(), 2:6] #no_fishx4
        no_tracks_i = np.size(kf, 0)
        no_tracks_avg += no_tracks_i
        if no_tracks_i:
            kf_pos = kf[:, :3]
            kf_phi = kf[:, 3]
            prot = data[i, 4*protagonist_id :  4*protagonist_id + 4] 
            prot_phi = prot[3]
            all_fish = np.array([data[i, 4*ii :  4*ii + 4] for ii in range(fishes) if ii != protagonist_id]) #for matching only us pos, no phi

            rel_pos_unrot = (all_fish[:, :3]- prot[:3])
            R = np.array([[math.cos(prot_phi), math.sin(prot_phi), 0],[-math.sin(prot_phi), math.cos(prot_phi), 0],[0,0,1]]) #rotate by phi around z axis to transform from global to robot frame
            groundtruth_pos = (R @ rel_pos_unrot.T).T
            groundtruth_phi = np.arctan2(np.sin(all_fish[:, 3] - prot_phi), np.cos(all_fish[:, 3] - prot_phi))
            
            dist = cdist(kf_pos, groundtruth_pos, 'euclidean')    
            kf_matched_ind, groundtruth_matched_ind = linear_sum_assignment(dist)
            error_i = dist[kf_matched_ind, groundtruth_matched_ind].sum()
            phi_diff = kf_phi[kf_matched_ind] - groundtruth_phi[groundtruth_matched_ind]
            error_phi = np.sum(abs(np.arctan2(np.sin(phi_diff), np.cos(phi_diff))))
            pos_tracking_error_avg += error_i/no_tracks_i
            phi_tracking_error_avg += error_phi/no_tracks_i

no_new_tracks_avg /= fishes
no_tracks_avg /= (fishes*timesteps)
print('Out of the {} fishes, an avg of {:0.1f} tracks were tracked per timestep. In avg {:0.1f} kf tracks were created during the whole experiment).'.format(fishes, no_tracks_avg, no_new_tracks_avg))


pos_tracking_error_avg /= (timesteps*fishes)
phi_tracking_error_avg /= (timesteps*fishes)

print('The avg pos tracking error is {:0.1f} mm and the avg phi tracking error is {:0.1f} deg.'.format(pos_tracking_error_avg, phi_tracking_error_avg*180/math.pi))

log_stat(experiment_type, loopname, filename, math.floor(timesteps/clock_freq), fishes, n_magnitude, surface_reflections, escape_angle, speed_ratio, phi_std[0], phi_std[-1], hull_area_max/1000**2, eaten, no_tracks_avg, no_new_tracks_avg, pos_tracking_error_avg/1000, phi_tracking_error_avg)
