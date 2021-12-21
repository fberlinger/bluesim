#!/usr/bin/python
"""Animates simulation data from logfiles with ipyvolume

Attributes:
    clock_freq (float): Clock frequency
    clock_rate (float): Clock rate
    colors (np-array of floats): Colors fish depending on their location
    fig (figure object): ipv figure
    fishes (int): Number of simulated fishes
    phi (float): Orientation angles
    quiver (plot object): ipv quiver plot
    timesteps (TYPE): Description
    v (float): Position magnitude
    x (float): x-positions
    y (float): y-positions
    z (float): z-positions
"""
import json
import numpy as np
import ipyvolume as ipv
import matplotlib.cm as cm
import copy
import sys


# Load Data
try:
    filename = sys.argv[1]
except:
    print('Provide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211')
    sys.exit()
try:
    data = np.loadtxt('./logfiles/{}.txt'.format(filename), delimiter=',')
    with open('./logfiles/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
except:
    print('Data file with prefix {} does not exist.\nProvide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    sys.exit()

# Read Experimental Parameters
clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = int(data.shape[1]/8)

# Format Data
x = data[:, :1]
y = data[:, 1:2]
z = data[:, 2:3]
phi = data[:, 3:4]
#vx = data[:, 4:5]
#vy = data[:, 5:6]
#vz = data[:, 6:7]

for ii in range(1,fishes):
    x = np.concatenate((x, data[:, 4*ii:4*ii+1]), axis=1)
    y = np.concatenate((y, data[:, 4*ii+1:4*ii+2]), axis=1)
    z = np.concatenate((z, data[:, 4*ii+2:4*ii+3]), axis=1)
    phi = np.concatenate((phi, data[:, 4*ii+3:4*ii+4]), axis=1)
    #vx = np.concatenate((vx, data[:, 4*(fishes+ii):4*(fishes+ii)+1]), axis=1)
    #vy = np.concatenate((vy, data[:, 4*(fishes+ii)+1:4*(fishes+ii)+2]), axis=1)
    #vz = np.concatenate((vz, data[:, 4*(fishes+ii)+2:4*(fishes+ii)+3]), axis=1)

# Colors
v = np.sqrt(x**2 + y**2 + z**2)
v -= v.min(); v /= v.max();
d = copy.deepcopy(z); d -= d.min(); d /= d.max();
colors = np.array([cm.Blues(k) for k in v])
#colors = np.array([cm.Blues(k) for k in d])
#colors[:, 0, :] = cm.Reds(0.5) # this fish is red

# Create Animation
fig = ipv.figure()
ipv.xlim(0, arena[0])
ipv.ylim(0, arena[1])
ipv.zlim(0, arena[2])
ipv.style.use('dark')

quiver = ipv.quiver(x, y, z, np.cos(phi), np.sin(phi), np.zeros((1,len(phi))), size=4, color=colors[:,:,:3], marker="sphere")
ipv.animation_control(quiver, interval=clock_rate)

ipv.save('./animations/{}_animation.html'.format(filename))

print('BLUEANIMAT saved your animation in ./animations/{}_animation.html.\nOpen with your favorite browser, sit back and enjoy the extravaganza!'.format(filename))