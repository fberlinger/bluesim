#!/usr/bin/python
"""Plots simulation data from logfiles with ipyvolume

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


plot_kf = True
plot_phi = True

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
pred_bool = meta['pred_bool']

# Format Data
# x = data[:, :1]
# y = data[:, 1:2]
# z = data[:, 2:3]
# phi = data[:, 3:4]


if plot_kf:
    protagonist_id = 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,15))

    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.axis('equal')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('phi [deg]')
    ax2.set_ylim([-180, 180])

    #add kalman filtered data if available (relative position)
    data_kf = np.genfromtxt('./logfiles/kf_{}.csv'.format(protagonist_id), delimiter=',')
    if data_kf.ndim == 2:
        data_kf = data_kf[1:,:] #cut title row
        data_kf_sort = data_kf[np.argsort(data_kf[:, 0])]
        tracks = np.unique(data_kf_sort[:,0]).astype(int)
        colors = cm.Blues(np.linspace(0.3,0.8,len(tracks)))

        for idx, ii in enumerate(tracks):
            data_kf_track = data_kf[np.argwhere(data_kf[:,0]==ii).ravel(), :]
            if ii == 1000: #pred (just a different label)
                 ax1.plot(data_kf_track[:,2], data_kf_track[:,3], marker = '*', markersize=12, label='kf pred', color = colors[idx,:])
                 ax2.plot(data_kf_track[:,1]/clock_freq, np.arctan2(np.sin(data_kf_track[:,5]), np.cos(data_kf_track[:,5])) * 180/np.pi, marker = '*', markersize=12, label='kf pred', color = colors[idx,:])
            else:
                ax1.plot(data_kf_track[:,2], data_kf_track[:,3], marker = '*', markersize=12, label='kf_{}'.format(ii), color = colors[idx,:])
                ax2.plot(data_kf_track[:,1]/clock_freq, np.arctan2(np.sin(data_kf_track[:,5]), np.cos(data_kf_track[:,5])) * 180/np.pi, marker = '*', markersize=12, label='kf_{}'.format(ii), color = colors[idx,:])
        
        ax2.scatter(data_kf[np.argwhere(data_kf[:, 8].astype(int)!=1),1]/clock_freq, data_kf[np.argwhere(data_kf[:, 8].astype(int)!=1), 5]* 180/np.pi, marker = '*', s=500, color = 'k')
        ax1.legend()
        ax2.legend()

    #add groundtruth relative position
    colors = cm.Reds(np.linspace(0.3,0.8,fishes+pred_bool))
    for ii in [ii for ii in range(fishes) if ii != protagonist_id]:
        rel_x_unrot = np.array(data[:, 4*ii]) - np.array(data[:, 4*protagonist_id])
        rel_y_unrot = np.array(data[:, 4*ii + 1]) - np.array(data[:, 4*protagonist_id + 1])
        rel_z = np.array(data[:, 4*ii + 2]) - np.array(data[:, 4*protagonist_id + 2])
        phi_prot = np.array(data[:, 4*protagonist_id + 3])
        phi_fish = np.array(data[:, 4*ii + 3])
        rel_x = rel_x_unrot * np.cos(phi_prot) + rel_y_unrot * np.sin(phi_prot)
        rel_y = - rel_x_unrot * np.sin(phi_prot) + rel_y_unrot * np.cos(phi_prot)
        rel_phi = np.arctan2(np.sin(phi_fish - phi_prot), np.cos(phi_fish - phi_prot))
        ax1.plot(rel_x, rel_y, marker = 'o', label='groundtruth_{}'.format(ii), color = colors[ii,:])
        ax2.plot(np.array(range(timesteps))/clock_freq, rel_phi * 180/np.pi, marker = 'o', label='groundtruth_{}'.format(ii), color = colors[ii,:])

    if pred_bool:
        pred_start = np.where(data[:-1, 8*fishes] != data[1:, 8*fishes])[0][0] + 1
        pred_end = np.where(data[:-1, 8*fishes] != data[1:, 8*fishes])[0][-1] + 1
        rel_x_unrot = np.array(data[pred_start:pred_end, 8*fishes]) - np.array(data[pred_start:pred_end, 4*protagonist_id])
        rel_y_unrot = np.array(data[pred_start:pred_end, 8*fishes + 1]) - np.array(data[pred_start:pred_end, 4*protagonist_id + 1])
        rel_z = np.array(data[pred_start:pred_end, 8*fishes + 2]) - np.array(data[pred_start:pred_end, 4*protagonist_id + 2])
        phi_prot = np.array(data[pred_start:pred_end, 4*protagonist_id + 3])
        phi_fish = np.array(data[pred_start:pred_end, 8*fishes + 3])
        rel_x = rel_x_unrot * np.cos(phi_prot) + rel_y_unrot * np.sin(phi_prot)
        rel_y = - rel_x_unrot * np.sin(phi_prot) + rel_y_unrot * np.cos(phi_prot)
        rel_phi = np.arctan2(np.sin(phi_fish - phi_prot), np.cos(phi_fish - phi_prot))
        ax1.plot(rel_x, rel_y, marker = 'o', label='groundtruth pred', color = colors[fishes,:])
        ax2.plot(np.array(range(pred_start, pred_end))/clock_freq, rel_phi * 180/np.pi, marker = 'o', label='groundtruth pred', color = colors[fishes,:])

    
    ax1.legend()
    ax2.legend()   

    fig.suptitle('Kf tracking of fish {}'.format(protagonist_id))
    plt.gcf().canvas.get_tk_widget().focus_force()  #apparently necessary for mac to open new figure in front of everything
    plt.show()
    #plt.get_current_fig_manager().show() #new figure in front of other figures, but behind spyder

"""plots all phi over time"""
if plot_phi:
    fig, axs = plt.subplots(1, 1, figsize=(15,15))

    phi_mean_cos = np.zeros((timesteps))
    phi_mean_sin = np.zeros((timesteps))

    for ii in range(fishes):
        axs.plot(np.array(range(timesteps))/clock_freq, np.arctan2(np.sin(data[:, 4*ii + 3]), np.cos(data[:, 4*ii + 3])), label=ii)
        phi_mean_cos += np.cos(data[:, 4*ii + 3])
        phi_mean_sin += np.sin(data[:, 4*ii + 3])

    phi_mean_cos = phi_mean_cos/fishes
    phi_mean_sin = phi_mean_sin/fishes
    phi_mean = np.arctan2(phi_mean_sin, phi_mean_cos)
    phi_std = np.sqrt(-np.log(phi_mean_sin**2 + phi_mean_cos**2))

    axs.plot(np.array(range(timesteps))/clock_freq, phi_mean, '--k', LineWidth=4, color ='gray', label="mean")
    axs.plot(np.array(range(timesteps))/clock_freq, phi_std, ':k', LineWidth=4, color ='gray', label="std")

    axs.set_xlabel('Time [s]')
    axs.set_ylabel('Phi [rad]')
    axs.legend()
    axs.set_ylim([-np.pi,np.pi])

    fig.suptitle('Phi over time')
    plt.show()

    print('The initial std phi is {0:.1f}rad.'.format(phi_std[0]))
    print('The final std phi is {0:.1f}rad.'.format(phi_std[-1]))
