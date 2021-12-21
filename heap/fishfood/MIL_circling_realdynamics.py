"""Simulates a Bluebot. Add behavioral code here.
"""
from math import *
import numpy as np
import random
import time

class Fish():
    """Bluebot instance
    """
    
    def __init__(self, my_id, dynamics, environment):
        # Arguments
        self.id = my_id
        self.dynamics = dynamics
        self.environment = environment

        # Bluebot features
        self.body_length = 160

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0

        # Behavior specific
        self.target_depth = random.randint(250, 300)


    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist = self.environment.get_robots(self.id)
        target_pos, vel = self.move(robots, rel_pos, dist, duration)
        self.environment.update_states(self.id, target_pos, vel)

    def depth_ctrl_vision(self, r_move_g):
        """Vision-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch_range = 2 # abs(pitch) below which dorsal fin is not controlled

        pitch = np.arctan2(r_move_g[2], sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / pi

        if pitch > pitch_range:
            self.dorsal = 1
        elif pitch < -pitch_range:
            self.dorsal = 0

    def depth_ctrl_psensor(self, r_move_g):
        """Pressure-sensor-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        depth = self.environment.pos[self.id,2]
        target_depth = depth + r_move_g[2]

        if depth > target_depth:
            self.dorsal = 1
        else:
            self.dorsal = 0

    def circling(self, robots, rel_pos):
        sensing_angle = 150 #deg

        if not robots:

            # 10 robots
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1
            '''
            # 20 robots
            self.pect_l = 0
            self.pect_r = 1.0
            self.caudal = 0.9
            '''

            #move = np.array([4, -1, 0])
            #move = np.array([4.09, -0.49, 0]) # 2r
            #move = np.array([4.12, -0.2, 0]) # 5r
            #return move
        
        #someone = self.environment.see_circlers(self.id, robots, rel_pos, sensing_angle)
        someone = self.environment.see_circlers_LoS(self.id, robots, rel_pos)

        if someone:

            # 10 robots
            self.pect_r = 0
            self.pect_l = 0.5
            self.caudal = 0.1
            '''
            # 20 robots
            self.pect_r = 0
            self.pect_l = 1.0
            self.caudal = 0.9
            '''

            #move = np.array([4, 1, 0])
            #move = np.array([4.09, 0.49, 0]) # 2r
            #move = np.array([4.12, 0.2, 0]) # 5r
        else:
            
            # 10 robots
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1
            '''
            # 20 robots
            self.pect_l = 0
            self.pect_r = 1.0
            self.caudal = 0.9
            '''            

            #move = np.array([4, -1, 0])
            #move = np.array([4.09, -0.49, 0]) # 2r
            #move = np.array([4.12, -0.2, 0]) # 5r

        #return move

    def move(self, robots, rel_pos, dist, duration):
        """Decision-making based on neighboring robots and corresponding move
        """
        #if not robots: # no robots, continue with ctrl from last step
        #    target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)
        #    return (target_pos, self_vel)

        # Define your move here
        self.circling(robots, rel_pos)
        #magn = np.linalg.norm(move) # normalize
        #move /= magn
        #print(move)

        # Global to Robot Transformation
        #phi = self.environment.pos[self.id,3]
        #r_T_g = self.environment.rot_global_to_robot(phi)
        #r_move_g = r_T_g @ move

        #self.depth_ctrl_vision(r_move_g)
        #self.home(r_move_g, magnitude)

        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)

        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        # POINT-MASS MOVE AT CONSTANT SPEED (0.5 BL PER ITERATION = 1 BL / S)
        '''
        phi = self.environment.pos[self.id,3]
        r_T_r = self.environment.rot_robot_to_global(phi)
        r_move_r = r_T_r @ move

        pos = self.environment.pos[self.id,:]
        pos[:3] += r_move_r * self.body_length/10
        pos[3] += np.arctan2(move[1], move[0])
        target_pos = pos #np.concatenate((pos, np.array([0])), axis=0)
        self_vel = np.array([0, 0, 0 ,0])
        '''

        return (target_pos, self_vel)