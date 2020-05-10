import math
import random
import time
import numpy as np
from scipy.spatial.distance import cdist

import sys

class Environment():
    """The dynamic network of robot nodes in the underwater environment

    This class keeps track of the network dynamics by storing the positions of
    all nodes. It contains functions to derive the distorted position from a
    target position by adding a distortion and noise, to update the position of
    a node, to update the distance between nodes, to derive the probability of
    receiving a message from a node based on that distance, and to get the
    relative position from one node to another node.
    """

    def __init__(self, pos, vel, arena):
        # Arguments
        self.pos = pos # x, y, z, phi; [no_robots X 4]
        self.vel = vel # pos_dot
        self.arena_size = arena # x, y, z
        
        # Parameters
        self.no_robots = self.pos.shape[0]
        self.no_states = self.pos.shape[1]

        # Initialize robot states
        self.init_states()

        # Initialize tracking
        self.init_tracking()

    def log_to_file(self, filename):
        np.savetxt('./logfiles/{}_data.txt'.format(filename), self.tracking, fmt='%.2f', delimiter=',')

    def init_tracking(self):
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))
        self.tracking = np.concatenate((pos,vel), axis=1)
        self.updates = 0

    def update_tracking(self):
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))
        current_state = np.concatenate((pos,vel), axis=1)
        self.tracking = np.concatenate((self.tracking,current_state), axis=0)
    
    def init_states(self):
        # Restrict initial positions to arena size
        self.pos[:,0] = np.clip(self.pos[:,0], 0, self.arena_size[0])
        self.pos[:,1] = np.clip(self.pos[:,1], 0, self.arena_size[1])
        self.pos[:,2] = np.clip(self.pos[:,2], 0, self.arena_size[2])

        # Initial relative positions
        a_ = np.reshape(self.pos, (1, self.no_robots*self.no_states))
        a = np.tile(a_, (self.no_robots,1))
        b = np.tile(self.pos, (1,self.no_robots))
        self.rel_pos = a - b # [4*no_robots X no_robots]

        # Initial distances
        self.dist = cdist(self.pos[:,:3], self.pos[:,:3], 'euclidean') # without phi; [no_robots X no_robots]

    def update_states(self, source_id, pos, vel): # add noise
        # Position and velocity
        self.pos[source_id,0] = np.clip(pos[0], 0, self.arena_size[0])
        self.pos[source_id,1] = np.clip(pos[1], 0, self.arena_size[1])
        self.pos[source_id,2] = np.clip(pos[2], 0, self.arena_size[2])
        self.pos[source_id,3] = pos[3]
        self.vel[source_id,:] = vel

        # Relative positions
        pos_others = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        pos_self = np.tile(self.pos[source_id,:], (1,self.no_robots))
        rel_pos = pos_others - pos_self
        self.rel_pos[source_id,:] = rel_pos # row
        rel_pos_ = np.reshape(rel_pos, (self.no_robots, self.no_states))
        self.rel_pos[:,source_id*self.no_states:source_id*self.no_states+self.no_states] = -rel_pos_ # columns
        
        # Relative distances
        dist = np.linalg.norm(rel_pos_[:,:3], axis=1) # without phi
        self.dist[source_id,:] = dist
        self.dist[:,source_id] = dist.T

        # Update status and tracking
        self.updates += 1
        if self.updates >= self.no_robots:
            self.updates = 0
            self.update_tracking()

    def get_robots(self, source_id, visual_noise=False):
        robots = set(range(self.no_robots)) # all robots
        robots.discard(source_id) # discard self

        rel_pos = np.reshape(self.rel_pos[source_id], (self.no_robots, self.no_states))

        self.visual_range(source_id, robots)
        self.blind_spot(source_id, robots, rel_pos)
        self.occlusions(source_id, robots, rel_pos)

        #if visual_noise: # no overwrites of self.rel_pos and self.dist
        #    n_rel_pos, n_dist = self.visual_noise(source_id, rel_pos)
        #    return (robots, n_rel_pos, n_dist)
        return (robots, rel_pos, self.dist[source_id])

    def visual_range(self, source_id, robots, v_range=3000):
        conn_drop = 0.005
        
        candidates = robots.copy()
        for robot in candidates:
            d_robot = self.dist[source_id][robot]
            x = conn_drop * (d_robot - v_range)
            if x < -5:
                sigmoid = 1
            elif x > 5:
                sigmoid = 0
            else:
                sigmoid = 1 / (1 + math.exp(x))
            prob = random.random()

            if  sigmoid < prob:
                robots.remove(robot)

    def blind_spot(self, source_id, robots, rel_pos, w_blindspot=50):
        """Omits robots within the blind spot behind own body.

        Args:
            source_id (int): Fish ID
            robots (set): Set of visible robots
            rel_pos (dict): Relative positions of robots
        """
        r_blockage = w_blindspot/2

        phi = self.pos[source_id,3]
        phi_xy = [math.cos(phi), math.sin(phi)]
        mag_phi = np.linalg.norm(phi_xy)
        
        candidates = robots.copy()
        for robot in candidates:
            dot = np.dot(phi_xy, rel_pos[robot,:2])
            if dot < 0:
                d_robot = np.linalg.norm(rel_pos[robot,:2])

                angle = abs(math.acos(dot / (mag_phi * d_robot))) - math.pi / 2 # cos(a-b) = ca*cb+sa*sb = sa

                if  math.cos(angle) * d_robot < r_blockage:
                    robots.remove(robot)

    def occlusions(self, source_id, robots, rel_pos, r_sphere=50):
        """Omits invisible robots occluded by others.

        Args:
            ource_id (int): Fish ID
            robots (set): Set of visible robots
            rel_pos (dict): Relative positions of robots
        """
        if not robots:
            return

        rel_dist = self.dist[source_id]
        id_by_dist = np.argsort(rel_dist)

        n_valid = []
        for robot in id_by_dist[1:]:
            if not robot in robots:
                continue
            occluded = False
            d_robot = rel_dist[robot]
            if d_robot == 0: # "collision"
                continue
            coord_robot = rel_pos[robot,:3]

            for verified in n_valid:
                d_verified = rel_dist[verified]
                #if d_verified == d_robot: # 2 neighbors in same location
                #    occluded = True
                #    robots.remove(robot)
                #    if not robots:
                #        return
                #    break

                coord_verified = rel_pos[verified,:3]

                theta_min = math.atan(r_sphere / d_verified)
                
                theta = abs(math.acos(np.dot(coord_robot, coord_verified) / (d_robot * d_verified)))

                if theta < theta_min:
                    occluded = True
                    robots.remove(robot)
                    if not robots:
                        return
                    break

            if not occluded:
                n_valid.append(robot)

    def visual_noise(self, source_id, rel_pos):
        magnitudes = 0.1 * np.array([self.dist[source_id]]).T # 10% of distance
        noise = magnitudes * (np.random.rand(self.no_robots, self.no_states) - 0.5) # zero-mean uniform noise
        n_rel_pos = rel_pos + noise
        n_dist = np.linalg.norm(n_rel_pos[:,:3], axis=1) # new dist without phi

        return (n_rel_pos, n_dist)

    def see_circlers(self, source_id, robots, rel_pos, sensing_angle):
        '''For circle formation
        '''
        phi = self.pos[source_id,3]
        phi_xy = [math.cos(phi), math.sin(phi)]
        mag_phi = np.linalg.norm(phi_xy)

        candidates = robots.copy()
        for robot in candidates:
            dot = np.dot(phi_xy, rel_pos[robot,:2])
            if dot > 0:
                d_robot = np.linalg.norm(rel_pos[robot,:2])

                angle = abs(math.acos(dot / (mag_phi * d_robot)))

                if (angle*180/math.pi) < (sensing_angle/2):
                    return True

        return False

    def rot_global_to_robot(self, phi):
            """Rotate global coordinates to robot coordinates. Used before simulation of dynamics.

            Args:
                source_id (id): Fish ID

            Returns:
                np.array: 3x3 rotation matrix based on current orientation
            """
            return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def rot_robot_to_global(self, phi):
            """Rotate global coordinates to robot coordinates. Used before simulation of dynamics.

            Args:
                source_id (id): Fish ID

            Returns:
                np.array: 3x3 rotation matrix based on current orientation
            """
            return np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])