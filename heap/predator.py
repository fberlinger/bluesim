"""Simulates the predator
"""
from math import *
import numpy as np
import time

from shapely.geometry import LineString, LinearRing #pip install necessary!

class Predator():
    """Predator instance
    """

    def __init__(self, dynamics, environment):
        # Arguments
        self.dynamics = dynamics
        self.environment = environment

        self.hunt_started = False
        self.deltat = 0.5 #pw dont hardcode!!!
        self.pred_max_speed = 30 #[mm/s] pw adapt
        self.pred_phi = 0

    def run(self):

        self.move()

    def calc_phi_mean_std(self):
        phi_array = np.vstack(self.environment.pos)[:,3]
        phi_mean_sin = 0
        phi_mean_cos = 0

        for value in phi_array:
            phi_mean_sin += np.sin(value)
            phi_mean_cos += np.cos(value)

        #calc mean and std
        phi_mean_cos = phi_mean_cos/np.size(phi_array, 0)
        phi_mean_sin = phi_mean_sin/np.size(phi_array, 0)
        phi_mean = np.arctan2(phi_mean_sin, phi_mean_cos)
        phi_std = np.sqrt(-np.log(phi_mean_sin**2 + phi_mean_cos**2))
        #print("fish_phi_mean",phi_mean, "fish_phi_std", phi_std)

        return phi_mean, phi_std

    def calc_pos_center(self):
        fish_pos = np.vstack(self.environment.pos)[:,:3]
        fish_center = np.mean(fish_pos, axis=0)

        return fish_center

    def start_pred(self, pred_phi, fish_center):
        fish_point1 = fish_center - np.array([cos(pred_phi), sin(pred_phi), 0])*10000 #approach from behind, just choose big enough factor to get out of tank for sure
        fish_path = LineString([tuple(fish_center[0:2]), tuple(fish_point1[0:2])])
        tank = LinearRing([(0, 0), (self.environment.arena_size[0], 0), (self.environment.arena_size[0], self.environment.arena_size[1]), (0, self.environment.arena_size[1])])

        intersect = fish_path.intersection(tank)
        print("predator started")

        self.environment.pred_pos[0] = intersect.x
        self.environment.pred_pos[1] = intersect.y
        self.environment.pred_pos[2] = fish_center[2]
        self.environment.pred_pos[3] = pred_phi

        self.environment.pred_visible = True

    def check_pred_visible(self):
        if self.environment.pred_pos[0] < 0 or self.environment.pred_pos[0] > self.environment.arena_size[0] or self.environment.pred_pos[1] < 0 or self.environment.pred_pos[1] > self.environment.arena_size[1] or self.environment.pred_pos[2] < 0 or self.environment.pred_pos[2] > self.environment.arena_size[2]:
            self.environment.pred_visible = False
            self.environment.pred_pos[0] = self.environment.arena_size[0]/2 #place predator outside of visible range
            self.environment.pred_pos[1] = -self.environment.arena_size[1]
            self.environment.pred_pos[2] = 0
            self.environment.pred_pos[3] = 0

    def simulate_predator_move(self, pred_speed, pred_phi):
        delta_dist = pred_speed*self.deltat
        pred_dir = np.array([np.cos(pred_phi), np.sin(pred_phi), 0])

        self.environment.pred_pos[:3] += delta_dist * pred_dir
        self.environment.pred_pos[3] = pred_phi
        self.check_pred_visible()

    def move(self):
        """Decision-making based on neighboring robots and corresponding move
        """
        pred_speed = 0

        if self.hunt_started:
            if self.environment.pred_visible:
                self.simulate_predator_move(self.pred_max_speed, self.pred_phi) #pw check dt

        else:
            fish_phi_mean, fish_phi_std = self.calc_phi_mean_std()
            std_thresh = 0.25
            #print(fish_phi_std)

            if fish_phi_std < std_thresh: #well enough aligned
                self.hunt_started = True
                self.pred_phi = fish_phi_mean
                fish_center = self.calc_pos_center()
                self.start_pred(self.pred_phi, fish_center)
