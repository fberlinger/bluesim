"""Central data base keeping track of positions, velocities, relative positions, and distances of all simulated fishes
"""
import math
import random
import numpy as np
from scipy.spatial.distance import cdist

U_LED_DX = 86 # [mm] leds x-distance on BlueBot
U_LED_DZ = 86 # [mm] leds z-distance on BlueBot

class Environment():
    """Simulated fish environment

    Fish get their visible neighbors and corresponding relative positions and distances from here. Fish also update their own positions after moving in here. Environmental tracking data is used for simulation analysis.
    """

    def __init__(self, pos, vel, fish_specs, arena, pred_bool, clock_freq):
        # Arguments
        self.pos = pos # x, y, z, phi; [no_robots X 4]
        self.vel = vel # pos_dot
        self.v_range = fish_specs[0] # visual range, [mm]
        self.w_blindspot = fish_specs[1] # width of blindspot, [mm]
        self.r_sphere = fish_specs[2] # radius of blocking sphere for occlusion, [mm]
        self.n_magnitude = fish_specs[3] # visual noise magnitude, [% of distance]
        self.surface_reflections = fish_specs[4] # boolean to activate surface reflections
        self.arena_size = arena # x, y, z
        self.pred_bool = pred_bool
        self.clock_freq = clock_freq

        if pred_bool:
            self.escape_angle = fish_specs[5] # escape angle for fish, [rad]
            self.pred_speed = fish_specs[6] # predator speed, [mm/s]

        self.fish_factor_speed = fish_specs[7] #slow down fish from max speed with this factor
        # Parameters
        self.no_robots = self.pos.shape[0]
        self.no_states = self.pos.shape[1]

        self.leds_pos = [np.zeros((3,3))]*np.size(self.pos,0) #empty init, filled with update_leds() below
        for i in range(np.shape(self.pos)[0]):
            self.update_leds(i)


        # Initialize robot states
        self.init_states()

        # Init predator
        if self.pred_bool:
            self.pred_pos = np.zeros((4)) #predator position x,y,z,phi
            self.pred_pos[0] = self.arena_size[0]/2
            self.pred_pos[1] = -self.arena_size[1] #place predator outside of visible range
            self.pred_visible = False

        # Initialize tracking
        self.init_tracking()


    def log_to_file(self, filename):
        """Logs tracking data to file
        """
        np.savetxt('./logfiles/{}_data.txt'.format(filename), self.tracking, fmt='%.2f', delimiter=',')

    def update_leds(self, source_index):
        #pos is led1
        pos = self.pos[source_index][0:3]
        phi = self.pos[source_index][3]

        x1 = pos[0]
        x2 = x1
        x3 = x1 + math.cos(phi)*U_LED_DX

        y1 = pos[1]
        y2 = y1
        y3 = y1 + math.sin(phi)*U_LED_DX

        z1 = pos[2]
        z2 = z1 + U_LED_DZ
        z3 = z1

        self.leds_pos[source_index] = np.array([[x1, x2, x3],[y1, y2, y3],[z1, z2, z3]])

    def init_tracking(self):
        """Initializes tracking
        """
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))

        if self.pred_bool:
            pred_pos = np.reshape(self.pred_pos, (1,self.no_states))
            self.tracking = np.concatenate((pos,vel,pred_pos), axis=1)
        else:
            self.tracking = np.concatenate((pos,vel), axis=1)

        self.updates = 0

    def update_tracking(self):
        """Updates tracking after every fish took a turn
        """
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))
        if self.pred_bool:
            pred_pos = np.reshape(self.pred_pos, (1,self.no_states))
            current_state = np.concatenate((pos,vel,pred_pos), axis=1)
        else:
            current_state = np.concatenate((pos,vel), axis=1)
        self.tracking = np.concatenate((self.tracking,current_state), axis=0)

    def init_states(self):
        """Initializes fish positions and velocities
        """
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
        """Updates a fish state and affected realtive positions and distances
        """
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

        # Update tracking
        self.updates += 1
        if self.updates >= self.no_robots:
            self.updates = 0
            self.update_tracking()

        # Update leds
        self.update_leds(source_id)

    def get_robots(self, source_id, visual_noise=False):
        """Provides visible neighbors and relative positions and distances to a fish
        """
        robots = set(range(self.no_robots)) # all robots
        robots.discard(source_id) # discard self

        rel_pos = np.reshape(self.rel_pos[source_id], (self.no_robots, self.no_states))

        self.visual_range(source_id, robots)
        self.blind_spot(source_id, robots, rel_pos)
        self.occlusions(source_id, robots, rel_pos)

        leds = self.calc_relative_leds(source_id, robots)

        if self.n_magnitude: # no overwrites of self.rel_pos and self.dist
            n_rel_pos, n_dist = self.visual_noise(source_id, rel_pos) #pw add leds noise here
            return (robots, n_rel_pos, n_dist, leds)

        return (robots, rel_pos, self.dist[source_id], leds)

    def visual_range(self, source_id, robots):
        """Deletes fishes outside of visible range
        """
        conn_drop = 0.005

        candidates = robots.copy()
        for robot in candidates:
            d_robot = self.dist[source_id][robot]
            x = conn_drop * (d_robot - self.v_range)
            if x < -5:
                sigmoid = 1
            elif x > 5:
                sigmoid = 0
            else:
                sigmoid = 1 / (1 + math.exp(x))
            prob = random.random()

            if  sigmoid < prob:
                robots.remove(robot)

    def blind_spot(self, source_id, robots, rel_pos):
        """Omits fishes within the blind spot behind own body
        """
        r_blockage = self.w_blindspot/2

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

    def occlusions(self, source_id, robots, rel_pos):
        """Omits invisible fishes occluded by others
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
                coord_verified = rel_pos[verified,:3]

                theta_min = math.atan(self.r_sphere / d_verified)
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
        """Adds visual noise
        """
        magnitudes = self.n_magnitude * np.array([self.dist[source_id]]).T # 10% of distance
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
            """
            return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def rot_robot_to_global(self, phi):
            """Rotate robot coordinates to global coordinates. Used after simulation of dynamics.
            """
            return np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def calc_reflections(self, leds_list):
        refl_list = []
        for led in leds_list:
            if led[2] > 10: #at least 10mm below surface to have a reflection
                refl = led + np.array([0,0, -2*led[2]])
                refl_list.append(refl)
        return refl_list

    def calc_relative_leds(self, source_id, robots):
        """Use right and left cameras just up to the xz-plane such that the overlapping camera range disappears and there are no duplicates.

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """
        all_blobs = np.empty((3,0))
        
        noisy_leds_pos = self.leds_pos.copy()
        if self.n_magnitude:
            for i in robots:
                magnitudes = self.n_magnitude * np.array([self.dist[source_id][i]]).T # 10% of distance
                noise = magnitudes * (np.random.rand(3, 1) - 0.5) # zero-mean uniform noise
                noisy_leds_pos[i] = noisy_leds_pos[i] + noise #add same noise on all 3 leds

        leds = [x for i,x in enumerate(noisy_leds_pos) if i in robots]   #ignore my own leds and only take leds of fish that I can see
        if leds:
            leds_list = list(np.transpose(np.hstack(leds)))

            if self.surface_reflections:
                refl_list = self.calc_reflections(leds_list)
                leds_list = leds_list + refl_list
            #print("leds_list",len(leds_list),leds_list)
            my_pos = self.pos[source_id][0:3]
            my_phi = self.pos[source_id][3]
            R = np.array([[math.cos(-my_phi), -math.sin(-my_phi), 0],[math.sin(-my_phi), math.cos(-my_phi), 0],[0,0,1]])# rotate into my coord system

            for led in leds_list:
                relative_coordinates = np.dot(R, ((led - my_pos)[:, np.newaxis]))
                relative_coordinates = relative_coordinates/np.linalg.norm(relative_coordinates)#normalize from xyz to pqr
                #add noise
                #noise_magnitude = 0#0.1 # +-10% of actual distance; scale noise with distance of point --> the further away, the less acurate is measurement
                #relative_coordinates += np.random.uniform(-1,1,3)[:, np.newaxis]*noise_magnitude #add noise
                all_blobs = np.append(all_blobs, relative_coordinates, axis=1)

        p = np.random.permutation(np.shape(all_blobs)[1]) #mix up the order to test sorting algorithm
        return all_blobs[:,p]

    def get_rel_pos_pred(self, source_index):
        """Calculate the relative position from the source node to the predator"""
        #rel pos
        global_heading = self.pred_pos[:3] - self.pos[source_index][:3]
        phi = self.pos[source_index][3]
        R = np.array([[math.cos(-phi), -math.sin(-phi), 0],[math.sin(-phi), math.cos(-phi), 0],[0,0,1]]) #rotate by phi around z axis to transform from global to robot frame
        robot_heading = R @ np.transpose(global_heading)

        #rel_phi
        diff = self.pred_pos[3] - phi
        rel_phi = np.arctan2(math.sin(diff), math.cos(diff))
        rel_pos_pred = np.append(robot_heading, rel_phi)
        return rel_pos_pred

    def perceive_pred(self, source_id, robots, rel_pos, rel_dist):
        pred_rel_pos = self.get_rel_pos_pred(source_id)

        #check if pred occluded by other fish
        occluded = False
        coord_robot = pred_rel_pos[:3]
        d_robot = np.linalg.norm(coord_robot)

        for verified in robots:
            d_verified = rel_dist[verified]
            coord_verified = rel_pos[verified,:3]

            theta_min = math.atan(self.r_sphere / d_verified)
            theta = abs(math.acos(np.dot(coord_robot, coord_verified) / (d_robot * d_verified)))

            if theta < theta_min and d_verified < d_robot:
                occluded = True
                #print(source_id, "predator is occluded")
                break

        #check if pred in blindspot
        blindspot = False
        r_blockage = self.w_blindspot/2

        phi = self.pos[source_id,3]
        phi_xy = [math.cos(phi), math.sin(phi)]
        mag_phi = np.linalg.norm(phi_xy)

        dot = np.dot(phi_xy, pred_rel_pos[:2])
        if dot < 0:
            d_robot = np.linalg.norm(pred_rel_pos[:2])
            angle = abs(math.acos(dot / (mag_phi * d_robot))) - math.pi / 2 # cos(a-b) = ca*cb+sa*sb = sa

            if math.cos(angle) * d_robot < r_blockage:
                blindspot = True
                #print(source_id, "predator in blindspot")

        #check if pred out of visual range
        out_of_visual_range = False
        conn_drop = 0.005
        d_robot =  np.linalg.norm(pred_rel_pos[:3])
        x = conn_drop * (d_robot - self.v_range)
        if x < -5:
            sigmoid = 1
        elif x > 5:
            sigmoid = 0
        else:
            sigmoid = 1 / (1 + math.exp(x))
        prob = random.random()

        if sigmoid < prob:
            out_of_visual_range = True

        detected = self.pred_visible and not (occluded or blindspot or out_of_visual_range)
        if not detected:
            pred_rel_pos = np.array([])

        return pred_rel_pos
