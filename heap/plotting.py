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
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


plot_kf = False
plot_phi = True

# Load Data
try:
    filename = sys.argv[1]
except:
    list_dir = os.listdir('logfiles')
    filename = sorted(list_dir)[-1][:13]
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
fishes = int(data.shape[1]/8)

# Format Data
# x = data[:, :1]
# y = data[:, 1:2]
# z = data[:, 2:3]
# phi = data[:, 3:4]


if plot_kf:
    protagonist_id = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,15))
    """
    #add non kalman filtered data (relative position)
    data = np.genfromtxt('nokf_{}.csv'.format(protagonist_id), delimiter=',')
    data = data[1:,:] #cut title row
    data_cut = np.delete(data, np.argwhere(data[:,5] == 0), axis = 0)
    data_sort = data_cut[np.argsort(data_cut[:, 0])]
    nr_tracks = int(data_sort[-1,0])

    #colors = cm.Greens(np.linspace(0.3,0.8,nr_tracks+1))
    for ii in range(nr_tracks+1):
        data_track = data_sort[np.argwhere(data_sort[:,0]==ii)[:,0], :]
        ax1.scatter(data_track[:,2], data_track[:,3], label='unfiltered_{}'.format(ii), marker = '*', s=250, color = 'g')
        ax2.scatter(data_track[:,1], data_track[:,5] * 180/np.pi, label='unfiltered_{}'.format(ii), marker = '*', s=250, color = 'g')
    ax1.legend()
    ax2.legend()
    """
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.axis('equal')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('phi [deg]')
    ax2.set_ylim([-180, 180])

    #add kalman filtered data if available (relative position)
    data = np.genfromtxt('kf_{}.csv'.format(protagonist_id), delimiter=',')
    if data.ndim == 2:
        data = data[1:,:] #cut title row
        data_cut = np.delete(data, np.argwhere(data[:,5] == 0), axis = 0)
        data_sort = data_cut[np.argsort(data_cut[:, 0])]
        tracks = np.unique(data_sort[:,0]).astype(int)
        colors = cm.Blues(np.linspace(0.3,0.8,len(tracks)))

        for idx, ii in enumerate(tracks):
            data_track = data[np.argwhere(data[:,0]==ii)[:,0], :]
            ax1.plot(data_track[:,2], data_track[:,3], marker = 'o', label='kf_{}'.format(ii), color = colors[idx,:])
            ax2.plot(data_track[:,1], np.arctan2(np.sin(data_track[:,5]), np.cos(data_track[:,5])) * 180/np.pi, marker = 'o', label='kf_{}'.format(ii), color = colors[idx,:])

        ax1.legend()
        ax2.legend()

    #add groundtruth relative position
    colors = cm.Reds(np.linspace(0.3,0.8,fishes))
    for ii in [ii for ii in range(fishes) if ii != protagonist_id]:
        rel_x_unrot = np.array(observer.x[ii]) - np.array(observer.x[protagonist_id])
        rel_y_unrot = np.array(observer.y[ii]) - np.array(observer.y[protagonist_id])
        rel_z = np.array(observer.z[ii]) - np.array(observer.z[protagonist_id])
        phi_prot = np.array(observer.phi[protagonist_id])
        phi_fish = np.array(observer.phi[ii])
        rel_x = rel_x_unrot * np.cos(phi_prot) + rel_y_unrot * np.sin(phi_prot)
        rel_y = - rel_x_unrot * np.sin(phi_prot) + rel_y_unrot * np.cos(phi_prot)
        rel_phi = np.arctan2(np.sin(phi_fish - phi_prot), np.cos(phi_fish - phi_prot))
        ax1.plot(rel_x, rel_y, marker = 'o', label='groundtruth_{}'.format(ii), color = colors[ii,:])
        ax2.plot(rel_phi * 180/np.pi, marker = 'o', label='groundtruth_{}'.format(ii), color = colors[ii,:])
        ax1.legend()
        ax2.legend()



    fig.suptitle('Kf tracking of fish {}'.format(protagonist_id))
    plt.gcf().canvas.get_tk_widget().focus_force()  #apparently necessary for mac to open new figure in front of everything
    plt.show()
    #plt.get_current_fig_manager().show() #new figure in front of other figures, but behind spyder

"""plots all phi over time"""
if plot_phi:

    fig, axs = plt.subplots(1, 1, figsize=(15,15))

    #axs.plot(observer.phi_mean[:], '--k', LineWidth=4, label='mean') pw calculate
    #axs.plot(observer.phi_std[:], ':k', LineWidth=4, label='std') pw calculate
    for ii in range(fishes):
        axs.plot(np.arctan2(np.sin(data[:, 8*ii+3 : 8*ii+4]), np.cos(data[:, 8*ii+3 : 8*ii+4])), label=ii)
    axs.set_xlabel('Time [s]')
    axs.set_ylabel('Phi [rad]')
    axs.legend()
    axs.set_ylim([-np.pi,np.pi])

    fig.suptitle('Phi over time')
    plt.show()

    #print('The initial std phi is {0:.1f}rad.'.format(observer.phi_std[0])) pw calculate
    #print('The final std phi is {0:.1f}rad.'.format(observer.phi_std[-1])) pw calculate
