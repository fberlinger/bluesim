from math import *
import numpy as np
import time


class Fish():
    def __init__(self, my_id, simulation_steps, dynamics, environment):
        # Arguments
        self.id = my_id
        self.simulation_steps = simulation_steps
        self.dynamics = dynamics
        self.environment = environment

        # Bluebot features
        self.body_length = 130
        self.w_blindspot = 50
        self.r_blocking = 65

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0

        self.is_started = False

    def start(self, lock):
        """Start the fish
        """
        self.is_started = True
        self.lock = lock
        self.steps = 0
        self.run()

    def run(self):
        while self.is_started:
            if self.steps >= self.simulation_steps:
                return

            with self.lock:
                robots_, rel_pos_, dist_ = self.environment.get_robots(self.id)
                robots = robots_.copy()
                rel_pos = rel_pos_.copy()
                dist = dist_.copy()
            target_pos, vel = self.move(robots, rel_pos, dist)
            with self.lock:
                self.environment.update_states(self.id, target_pos, vel)

            self.steps += 1

    def lj_force(self, robots, rel_pos, dist, r_target):
        """lj_force derives the Lennard-Jones potential and force based on the relative positions of all neighbors and the desired self.target_dist to neighbors. The force is a gain factor, attracting or repelling a fish from a neighbor. The center is a point in space toward which the fish will move, based on the sum of all weighted neighbor positions.

        Args:
            neighbors (set): Visible neighbors
            rel_pos (dict): Relative positions of visible neighbors

        Returns:
            np.array: Weighted 3D direction based on visible neighbors
        """
        a = 12
        b = 6
        epsilon = 1 # depth of potential well, V_LJ(r_target) = epsilon
        gamma = 10 # force gain
        r_const = r_target + 2 * self.body_length

        center = np.zeros((3,))
        n = len(robots)

        for robot in robots:
            r = min(dist[robot], r_const)
            f_lj = -gamma*epsilon/r * (a*(r_target/r)**a - 2*b*(r_target/r)**b)
            center += f_lj * rel_pos[robot,:3]

        center /= n
        magn = np.linalg.norm(center) # normalize
        center /= magn # normalize

        return (center, magn)

    def depth_ctrl_vision(self, r_move_g):
        """Controls diving depth based on direction of desired move.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch_range = 1 # abs(pitch) below which dorsal fin is not controlled

        pitch = np.arctan2(r_move_g[2], sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / pi

        if pitch > pitch_range:
            self.dorsal = 1
        elif pitch < -pitch_range:
            self.dorsal = 0

    def depth_ctrl_psensor(self, r_move_g):
        """Controls diving depth in a pressure sensor fashion. Own depth is "measured", i.e. reveiled by the interaction. Depth control is then done based on a target depth coming from a desired goal location in the robot frame.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        depth = self.environment.pos[self.id,2]
        target_depth = depth + r_move_g[2]

        if depth > target_depth:
            self.dorsal = 1
        else:
            self.dorsal = 0

    def home(self, r_move_g, magnitude):
        """Homing behavior. Sets fin controls to move toward a desired goal location.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        caudal_range = 35 # abs(heading) below which caudal fin is switched on
        freq_c = min(0.5 + 1/250 * magnitude, 1)

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / pi

        # target behind
        if heading > 155 or heading < -155:
            self.caudal = 0
            self.pect_r = 1.5
            self.pect_l = 1.5

        # target in front
        elif heading < 10 and heading > -10:
            self.pect_r = 0
            self.pect_l = 0
            self.caudal = freq_c

        # target to the right
        elif heading > 10:
            freq_l = 0.5 + 1 * abs(heading) / 155
            self.pect_l = freq_l
            self.pect_r = 0

            if heading < caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

        # target to the left
        elif heading < -10:
            freq_r = 0.5 + 1 * abs(heading) / 155
            self.pect_r = freq_r
            self.pect_l = 0

            if heading > -caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

    def move(self, robots, rel_pos, dist):
        if not robots: # no robots, continue with ctrl from last step
            target_pos, self_vel = self.dynamics.simulate_move(self.id)
            return (target_pos, self_vel)

        centroid_pos, magnitude = self.lj_force(robots, rel_pos, dist, r_target=390)
        move = centroid_pos

        # Global to Robot Transformation
        phi = self.environment.pos[self.id,3]
        r_T_g = self.environment.rot_global_to_robot(phi)
        r_move_g = r_T_g @ move

        self.depth_ctrl_vision(r_move_g)
        self.home(r_move_g, magnitude)

        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)

        target_pos, self_vel = self.dynamics.simulate_move(self.id)

        return (target_pos, self_vel)