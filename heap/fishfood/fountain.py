"""Fountain maneuver
"""
from math import *
import numpy as np
import time
import os, glob

from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

U_LED_DX = 86 # [mm] leds x-distance on BlueBot
U_LED_DZ = 86 # [mm] leds z-distance on BlueBot

class Fish():

    def __init__(self, my_id, dynamics, environment):
        # Arguments
        self.id = my_id
        self.dynamics = dynamics
        self.environment = environment

        # Bluebot features
        self.body_length = 130

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0

        # behavior specific parameters
        self.it_counter = 0
        self.behavior = 'align' #'aggregate'#pw change back
        self.escaped = False
        self.pred_undetected_count = 0
        self.escape_iteration_count = 0
        self.angle_escape = []

        #kf tracker init, move this to init
        self.kf_array = []
        self.first_detection = np.empty(0, dtype=int)
        self.last_detected = np.empty(0, dtype=int)
        self.track_id = np.empty(0, dtype=int)
        self.next_track_id = 0
        self.kf_phi_rotations = np.empty(0, dtype=int)
        self.kf_phi_prev = np.empty(0)


        # Logger instance
        for filename in glob.glob("./logfiles/kf*"): #remove all previous files
            os.remove(filename)

        with open('./logfiles/kf_{}.csv'.format(self.id), 'w') as f:
            f.truncate()
            f.write('TRACK_ID, ITERATION, X, Y, Z, PHI \n')

    def init_kf(self, xyz_init, phi_init):
        dt = 0.5 #[s]#pw measure this!!!
        dim_state = 8
        dim_meas = 4 #only measure pos, not vel
        #dim_input = dim_state - dim_meas #only vel
        noise_process_xyz = 50 #[mm]
        noise_process_phi = np.pi/8 #[rad]
        noise_process_vxyz = 100 #[mm/s]
        noise_process_vphi = np.pi/4 #[rad/s]
        noise_meas_xyz = 100 #[mm]
        noise_meas_phi = np.pi/4 #[rad]
        covar_init_xyz = 2000 #very unsure at the beginning -->large variance
        covar_init_phi = np.pi #very unsure at the beginning -->large variance
        covar_init_vxyz = 2000 #very unsure at the beginning -->large variance
        covar_init_vphi = np.pi #very unsure at the beginning -->large variance
        v_init = np.zeros((4)) #zeros velocity at the beginning

        kf = KalmanFilter(dim_x=dim_state, dim_z=dim_meas)#, dim_u=dim_input)
        kf.x = np.concatenate((xyz_init, phi_init, v_init), axis=None)[:,np.newaxis]
        kf.F = np.identity(dim_state) + np.pad(dt*np.identity(dim_meas), ((0, 4), (4, 0)), 'constant',constant_values = (0,0)) #transition matrix: assume const vel; PW in BlueSwarm code should F be changed every iteration because dt is not const? Time-varying kalman filter
        kf.H = np.pad(np.identity(dim_meas), ((0, 0), (0, 4)), 'constant',constant_values = (0,0)) #measurement matrix: we measure pos
        #kf.B = np.append(np.zeros((4,4)), np.identity(dim_input), axis=0) #control matrix: u is directy imu input vel (integrated acc) and v_phi (gyro), so B is zero on position and identity on velocity
        kf.R = np.diag([noise_meas_xyz, noise_meas_xyz, noise_meas_xyz, noise_meas_phi]) #measurement noise
        kf.Q = np.diag([noise_process_xyz, noise_process_xyz, noise_process_xyz, noise_process_phi, noise_process_vxyz, noise_process_vxyz, noise_process_vxyz, noise_process_vphi]) #process noise #Q_discrete_white_noise(dim=dim_state, dt=dt, var=var_state)
        kf.P = np.diag([covar_init_xyz, covar_init_xyz, covar_init_xyz, covar_init_phi, covar_init_vxyz, covar_init_vxyz, covar_init_vxyz, covar_init_vphi]) #estimated initial covariance matrix (a measure of the estimated accuracy of the state estimate)

        # add to arrays for tracking kf variables
        self.last_detected = np.append(self.last_detected, 0)
        self.first_detection = np.append(self.first_detection, 1)
        self.kf_phi_rotations = np.append(self.kf_phi_rotations, 0)
        self.kf_phi_prev = np.append(self.kf_phi_prev, 0)
        self.track_id = np.append(self.track_id, self.next_track_id)
        self.next_track_id += 1

        return kf

    def init_kf_pred(self, alpha_init, beta_init):
        dt = 0.5 #[s]
        dim_state = 4 #state contains for elements: alpha, beta, valpha, vbeta
        dim_meas = 2 #only measure pos, not vel
        dim_input = dim_state - dim_meas #only vel
        noise_process_alpha = np.pi/8 #[rad]
        noise_process_beta = np.pi/8 #[rad]
        noise_process_valpha = np.pi/4 #[rad/s]
        noise_process_vbeta = np.pi/4 #[rad/s]
        noise_meas_alpha = np.pi/4 #[rad]
        noise_meas_beta = np.pi/4 #[rad]
        covar_init_alpha = np.pi #very unsure at the beginning -->large variance
        covar_init_beta = np.pi #very unsure at the beginning -->large variance
        covar_init_valpha = np.pi #very unsure at the beginning -->large variance
        covar_init_vbeta = np.pi #very unsure at the beginning -->large variance
        v_init = np.zeros((dim_input)) #zeros velocity at the beginning

        kf = KalmanFilter(dim_x=dim_state, dim_z=dim_meas, dim_u=dim_input)
        kf.x = np.concatenate((alpha_init, beta_init, v_init), axis=None)[:,np.newaxis]
        kf.F = np.identity(dim_state) + np.pad(dt*np.identity(dim_meas), ((0, dim_state-dim_meas), (dim_state-dim_meas, 0)), 'constant',constant_values = (0,0)) #transition matrix: assume const vel; PW in BlueSwarm code should F be changed every iteration because dt is not const? Time-varying kf
        kf.H = np.pad(np.identity(dim_meas), ((0, 0), (0, dim_state-dim_meas)), 'constant',constant_values = (0,0)) #measurement matrix: we measure pos
        kf.B = np.append(np.zeros((dim_state-dim_input,dim_state-dim_input)), np.identity(dim_input), axis=0) #control matrix: u is directy imu input vel (integrated acc) and v_phi (gyro), so B is zero on position and identity on velocity
        kf.R = np.diag([noise_meas_alpha, noise_meas_beta]) #measurement noise
        kf.Q = np.diag([noise_process_alpha, noise_process_beta, noise_process_valpha, noise_process_vbeta]) #process noise #Q_discrete_white_noise(dim=dim_state, dt=dt, var=var_state)
        kf.P = np.diag([covar_init_alpha, covar_init_beta, covar_init_valpha, covar_init_vbeta]) #estimated initial covariance matrix (a measure of the estimated accuracy of the state estimate)

        return kf

    def start(self):
        """Start the process

        This sets `is_started` to true and invokes `run()`.
        """
        self.is_started = True
        self.run()

    def stop(self):
        """Stop the process

        This sets `is_started` to false.
        """
        self.is_started = False

    def log_kf(self, rel_pos, rel_orient):
        """Log current state
        """
        with open('./logfiles/kf_{}.csv'.format(self.id), 'a+') as f:
            for i in range(len(rel_pos)):
                f.write(
                    '{}, {}, {}, {}, {}, {}\n'.format(
                        self.track_id[i],
                        self.it_counter,
                        rel_pos[i][0],
                        rel_pos[i][1],
                        rel_pos[i][2],
                        rel_orient[i]
                    )
                )

    def run(self):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        target_pos, vel = self.move(leds)
        self.environment.update_states(self.id, target_pos, vel)

    def kalman_prediction_update(self):
        nr_tracks = len(self.kf_array)

        #if there are no tracks, create a new one
        if not nr_tracks:
            kf_new = self.init_kf(np.array([[0],[0],[0]]), np.array([0]))
            self.kf_array.append(kf_new)
            nr_tracks = 1

        #prediction step
        for i in range(nr_tracks):
            self.kf_array[i].predict()

        #calc all led positions
        xyz_threeblob_all = []
        rel_orient = []
        for i in range(nr_tracks):
            predicted_state = self.kf_array[i].x
            xyz_led1 = predicted_state[0:3]
            phi_bounded = np.arctan2(sin(predicted_state[3]), cos(predicted_state[3]))
            rel_orient.append(phi_bounded)
            xyz_threeblob = self.calc_all_led_xyz(xyz_led1, rel_orient[i])
            xyz_threeblob_all.append(xyz_threeblob)

        return xyz_threeblob_all, rel_orient

    def kalman_measurement_update(self, xyz, phi, track_ind):
        for i,j in zip(track_ind, range(len(xyz))):
            # angles inside kf are not restricted to (-pi, pi) but continous -->add up rotations
            if phi[j] - self.kf_phi_prev[i] < -pi:
                self.kf_phi_rotations[i] += 1
            elif phi[j] - self.kf_phi_prev[i] > pi:
                self.kf_phi_rotations[i] -= 1
            self.kf_phi_prev[i] = phi[j]
            phi[j] += self.kf_phi_rotations[i]*2*pi

            measured_state = np.append(xyz[j][:,0], [phi[j]], axis = 0)
            self.kf_array[i].update(measured_state)
            self.last_detected[i] = 0

    def kalman_remove_lost_tracks(self, track_ind):
        #delete all tracks which haven't been detected for too long
        nr_tracks = len(self.kf_array)
        not_detected = list(set(range(nr_tracks)).difference(set(track_ind)))

        self.last_detected[not_detected] += 1 #add 1 to all tracks which haven't been detected in this iteration
        lost_fish = np.array(self.last_detected) > 2 #if not detected in the last 3 frames, delete track

        outlier_fish = np.logical_and(self.first_detection, np.array(self.last_detected) > 0) #outlier fish are those which have been detected once for the first time and the next time not anymore
        self.first_detection[:nr_tracks] = 0

        delete_track_ind = np.argwhere(np.logical_or(lost_fish, outlier_fish))
        self.kf_array = [i for j, i in enumerate(self.kf_array) if j not in delete_track_ind]
        self.last_detected = np.delete(self.last_detected, delete_track_ind)
        self.first_detection = np.delete(self.first_detection, delete_track_ind)
        self.track_id = np.delete(self.track_id, delete_track_ind)
        self.kf_phi_prev = np.delete(self.kf_phi_prev, delete_track_ind)
        self.kf_phi_rotations = np.delete(self.kf_phi_rotations, delete_track_ind)


    def kalman_new_tracks(self, xyz_new, phi_new):
        nr_tracks = len(self.kf_array)
        # append new detections
        for j in range(len(phi_new)):
            #print("newtrack",xyz_new[j][:,0])
            self.kf_array.append(self.init_kf(xyz_new[j][:,0], phi_new[j]))

        #read out results
        rel_pos_led1 = []
        rel_phi = []

        for i in range(nr_tracks):#range(len(self.kf_array)):  the newest measurements will be ignored to prevent outliers
            estimated_state = self.kf_array[i].x
            xyz_led1 = estimated_state[0:3].ravel()
            rel_pos_led1.append(xyz_led1)
            phi_bounded = np.arctan2(sin(estimated_state[3]), cos(estimated_state[3]))
            rel_phi.append(phi_bounded)
        #print("rel_pos_led1", rel_pos_led1)
        self.log_kf(rel_pos_led1, rel_phi)
        return (rel_pos_led1, rel_phi)

    def kalman_tracking_predator(self, alpha_detection, beta_detection):
        # vel = np.zeros(2)#self.interaction.perceive_vel(self.id) #in imu we only get acc, this is already integrated once!
        # movement = np.append(vel, v_phi)[:,np.newaxis]

        #prediction step
        self.kf_pred.predict()

        #measurements
        if not alpha_detection == []:
            measured_state = np.append(alpha_detection, beta_detection)
            self.kf_pred.update(measured_state)

        return self.kf_pred.x[0], self.kf_pred.x[1], self.kf_pred.x[2], self.kf_pred.x[3]#alpha, beta, valpha, vbeta

    def aggregate(self, center_pos):
        phi_des = np.arctan2(center_pos[1], center_pos[0])
        v_des = 0.02 * (1 - abs(phi_des)/pi)#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
        dist_thresh = 400 #mm
        dist_center = center_pos[0]**2 + center_pos[1]**2
        #print(self.id, "dist_center",dist_center)
        if dist_center < dist_thresh**2:
            print(self.id, "aligning")
            self.behavior = "align"
        return (phi_des, v_des)

    def align(self, center_orient, pred_alpha, pred_beta, predator_detected):
        v_des = 0
        phi_des = center_orient
        if predator_detected and abs(pred_alpha)>pi/2 and not self.escaped: #if predator is visible and behind me and I havent already done a previous escape maneuver
            self.behavior = "predator_escape"
            print(self.id, "predator_escape")
            self.escaped = True
            #angle_sum =  pred_alpha - center_orient #correct my own perspective by my orientation towards swarm pw problem is that swarm is already swimming away --> center_orient not meaningful
            if pred_alpha > 0:#np.arctan2(sin(angle_sum), cos(angle_sum)) > 0:
                self.angle_escape = 110 /180*pi #120
            else:
                self.angle_escape = -110 /180*pi #120

            # init predator kf
            self.kf_pred = self.init_kf_pred(pred_alpha, pred_beta)

        return (phi_des, v_des)

    def predator_escape(self, center_pos, center_orient, pred_alpha, predator_detected):
        self.escape_iteration_count += 1

        if not predator_detected: #pw change this?
            phi_des = 0 #wait till we see it again; if not after 5 times, change to aggregation
            v_des = 0
            self.pred_undetected_count += 1
            if self.pred_undetected_count > 3:
                self.behavior = "aggregate_and_turn"
                print(self.id, "aggregate_and_turn cause pred not seen")
        else:
            self.pred_undetected_count = 0

            angle_pred_escape = pred_alpha + self.angle_escape
            angle_aggregation = np.arctan2(center_pos[1], center_pos[0])
            weight_escape = 1 #0.7

            angle_sum = weight_escape*angle_pred_escape + (1-weight_escape)*angle_aggregation
            phi_des = np.arctan2(sin(angle_sum), cos(angle_sum))
            v_des = 0.02 * min(1, (2 - 2*abs(phi_des)/pi))#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
            #PW DOES THIS MAKE SENSE?
            dist_thresh = 500 #mm
            dist_center = center_pos[0]**2 + center_pos[1]**2
            #print("dist_center",dist_center)
            if dist_center < dist_thresh**2 and dist_center and self.escape_iteration_count > 60: #if Im close to my friends and i've been escaping already for some time
                print(self.id, "aggregate_and_turn cause at dist:", sqrt(dist_center))
                self.behavior = "aggregate_and_turn"
            # phi_thresh = 150 /180*pi
            # if not predator_detected: #abs(pred_rel_phi) > phi_thresh: #or  need tracking to say which way to turn then #pw phi might be hard to determine at all time, use tracking! HOW? CURRENTLY JUST IF I DONT SEE IT ANYMORE THEN REAGGREGATE, otherwise keep swimming away from it DOESNT MAKE SENSE!
            #     print(self.id, "aggregate_and_turn")
            #     self.behavior = "aggregate_and_turn"

        return (phi_des, v_des)

    def aggregate_and_turn(self, center_pos):
        phi_des = np.arctan2(center_pos[1], center_pos[0])
        v_des = 0.02 * (1 - abs(phi_des)/pi)#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
        phi_aggregating = np.arctan2(center_pos[1], center_pos[0]) #assuming that fish also see their friends on the other side of fountain
        phi_turning = -self.angle_escape/2
        a = 0.5 #weight aggregation; pw tune
        b = 1 - a #weight turning
        phi_des = a*phi_aggregating + b*phi_turning

        dist_thresh = 200 #mm
        dist_center = center_pos[0]**2 + center_pos[1]**2
        #print("dist_center",dist_center)
        if dist_center < dist_thresh**2:
            print(self.id, "aligning")
            self.behavior = "align"

        return (phi_des, v_des)

    def comp_center_orient(self, rel_orient): # pw added
        """Compute the (potentially weighted) centroid of the fish neighbors orientation

        Arguments:
            rel_orient {dict} -- Dictionary of relative orientation to the
                neighboring fish.

        Returns:
            np.array -- angle centroid
        """
        orient_sin = 0
        orient_cos = 0
        orientation = 0

        for value in rel_orient:
            orient_sin += np.sin(value) # pw weighted with distance?
            orient_cos += np.cos(value)

        orientation = np.arctan2(orient_sin, orient_cos)

        return orientation

    def comp_center_pos(self, rel_pos):
        """Compute the (potentially weighted) centroid of the fish neighbors

        Arguments:
            rel_pos {dict} -- Dictionary of relative positions to the
                neighboring fish.

        Returns:
            np.array -- 3D centroid
        """
        center = np.zeros((3,))
        n = max(1, len(rel_pos))
        for value in rel_pos: # pw weighted with distance?
            center += value

        center /= n

        return center


    def depth_ctrl(self, r_move_g):
        """Controls diving depth based on direction of desired move.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch = np.arctan2(r_move_g[2], sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / pi

        if pitch > 1:
            self.dorsal = 1
        elif pitch < -1:
            self.dorsal = 0

    def depth_ctrl_vert(self, z_des):
        """Controls diving depth in a pressure sensor fashion, based on a target depth coming from a desired goal location in the robot frame.

        Args:
            z_des: Relative position of desired goal z coordinate in robot frame.
        """
        dead_band = 20
        #print("z_des", z_des)
        if z_des > dead_band:
            self.dorsal = min(1, 0.35 + z_des/200) #pw add d-part to p controller? check useful values
        else:
            self.dorsal = 0.35 #pw what is value to hold depth?

    def calc_all_led_xyz(self, xyz_led1, rel_orient):
        """Calculates the xyz coordinates of all three blobs based on xyz of first led and rel angle

        Args:
        xyz_led1 (float array 3x1): xyz coordinates of the first led of observed fish
        rel_orient (float): relative orientation of observed fish

        Returns:
        xyz_threeblob (float array 3x3): angle phi
        """
        xyz_led2 = xyz_led1 + np.array([0, 0, U_LED_DZ])[:,None]

        xyz_led3 = xyz_led1 + np.array([cos(rel_orient)*U_LED_DX, sin(rel_orient)*U_LED_DX, 0])[:,None]

        xyz_all = np.column_stack((xyz_led1, xyz_led2, xyz_led3))

        return xyz_all

    def _orientation(self, xyz): #adapted from BlueSwarm code
        """Calculates the orientation angle phi using the xyz coordinates

        Args:
        xyz (float array 3x3): xyz coordinates of the three blobs (sorted)

        Returns:
        float: angle phi
        """

        phi = np.arctan2(xyz[1,2]-xyz[1,0], xyz[0,2]-xyz[0,0])

        return phi

    def _pqr_to_xyz(self, pqr): #twoblob
        p1 = pqr[0, 0]
        q1 = pqr[1, 0]
        r1 = pqr[2, 0]
        p2 = pqr[0, 1]
        q2 = pqr[1, 1]
        r2 = pqr[2, 1]

        if r2 < r1:
            ptemp = p1
            qtemp = q1
            rtemp = r1
            p1 = p2
            q1 = q2
            r1 = r2
            p2 = ptemp
            q2 = qtemp
            r2 = rtemp

        delta = U_LED_DZ

        xyz = np.empty([3,2])
        if abs(r2*p1 -r1*p2) < 0.0001:
            #print("pqr div by zero risk",pqr)
            d1 = 1
        else:
            d1 = p2 * delta/(r2*p1 -r1*p2)

        if abs(r2) < 0.0001:
            # print("pqr div by zero risk",pqr)
             d2 = d1
        else:
             d2 = (d1*r1 + delta)/r2

        xyz[:,0] = d1 * pqr[:,0]
        xyz[:,1] = d2 * pqr[:,1]

        return xyz

    def _pqr_3_to_xyz(self, xyz_1_2, pqr_3):
        """Converts blob3 from pqr into xyz by finding the scale factor
        Args:
        pqr_3 (float array 3x1): pqr coordinates of blob3
        xyz_1_2 (float array 3x2): xyz coordinates of blob1,2

        Returns:
        xyz (float array 3x3): xyz coordinates of the three blobs
        """

        x1 = xyz_1_2[0, 0]
        y1 = xyz_1_2[1, 0]
        z1 = xyz_1_2[2, 0]
        p3 = pqr_3[0]
        q3 = pqr_3[1]
        r3 = pqr_3[2]

        delta = U_LED_DX

        a = p3**2 + q3**2
        b = -2 * (x1*p3 + y1*q3)
        c = x1**2 + y1**2 - delta**2

        sqrt_fix = max(b**2 - 4 * a * c, 0) #pw preventing negative sqrt, if it is too far off, the blob will be discarded later by led_hor_dist
        d_plus = (-b + sqrt(sqrt_fix)) / (2 * a)
        d_minus = (-b - sqrt(sqrt_fix)) / (2 * a)

        diff_z_plus = abs(z1 - d_plus*pqr_3[2])
        diff_z_minus = abs(z1 - d_minus*pqr_3[2])

        if (diff_z_plus < diff_z_minus):
            xyz_3 = d_plus * pqr_3
        else:
            xyz_3 = d_minus * pqr_3

        xyz = np.append(xyz_1_2, xyz_3[:,np.newaxis], axis=1)

        return xyz

    def calc_relative_angles(self, all_blobs): #copied and adapted from BlueSwarm Code "avoid_duplicates_by_angle" #pw split this up in env and fish part?
        """Use right and left cameras just up to the xz-plane such that the overlapping camera range disappears and there are no duplicates.

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """
        all_angles = np.empty(0)
        for i in range(np.shape(all_blobs)[1]):
            led = all_blobs[:,i]
            angle = np.arctan2(led[1], led[0])
            all_angles = np.append(all_angles, angle)

        return all_angles #angles in rad!

    def parse_orientation_reflections(self, all_blobs, predicted_blobs, predicted_phi):
        """Assigns triplets of blobs to single robots

        Idea: Sort all blobs by the angles/directions they are coming from. Pair duos of blobs that have most similar angles add third led.

        Args:
            all_blobs (np.array): all valid blobs from both images
            all_angles (np.array): all angles in xy-plane of all valid blobs

        Returns:
            tuple: set of neighbors and dict with their relative positions
        """
        xyz_twoblob_candidate = []
        twoblob_candidate_ind = []
        xyz_threeblob_matched = []
        xyz_threeblob_matched_ind = []
        track_threeblob_matched_ind = []
        phi_matched = []
        phi_new = []
        xyz_threeblob_new = []
        xyz_threeblob_new_ind = []

        nr_blobs = np.shape(all_blobs)[1]
        if nr_blobs < 3: # 0 robots
            return (xyz_threeblob_matched, phi_matched, xyz_threeblob_new, phi_new, track_threeblob_matched_ind)

        # find all valid blobs and their respective angles
        all_angles = self.calc_relative_angles(all_blobs)

        # subfunction: find vertically aligned leds (xyz_twoblob_candidates) and sort out reflections where >2 blobs have similar angle
        sorted_indices = np.argsort(all_angles)
        unassigned_ind = set(range(nr_blobs))
        angle_thresh = 3 / 180*np.pi # below which 2 blobs are considered a duo (pw make 5deg, smaller for testing)
        vert_thresh = 0.0001 #pqr normalized

        i = 0 # blob_ind
        neighbor_ind = 0
        while i+1 < nr_blobs: # iterate through all blobs and fill array
            # if 2 blobs are too far apart, ignore first one and check next 2
            dangle = abs(all_angles[sorted_indices[i+1]] - all_angles[sorted_indices[i]])
            vert_dist = abs(all_blobs[2, sorted_indices[i+1]] - all_blobs[2, sorted_indices[i]]) #check that vertically aligned blobs have a certain min distance
            #print("vert_dist",vert_dist)
            if not(dangle < angle_thresh and vert_dist > vert_thresh):
                if dangle < angle_thresh and vert_dist < vert_thresh: #assume blobs are identical if the angles and vert distance is close together --> discard one of them
                    unassigned_ind.remove(sorted_indices[i])
                i += 1
                continue

            # else, add 2 blobs
            ind_1 = sorted_indices[i]
            ind_2 = sorted_indices[i+1]
            b1 = all_blobs[:,ind_1]
            b2 = all_blobs[:,ind_2]

            # check for reflections
            ref = 0
            if i+2 < nr_blobs:
                dangle = abs(all_angles[sorted_indices[i+2]] - all_angles[sorted_indices[i]])
                vert_dist = abs(all_blobs[2,sorted_indices[i+2]] - all_blobs[2,sorted_indices[i]]) #check that vertically aligned blobs have a certain min distance
                if dangle < angle_thresh and vert_dist > vert_thresh: # a 3rd blob from same direction?
                    ref = 1
                    ind_3 = sorted_indices[i+2]
                    b3 = all_blobs[:,ind_3]
                    # who is closest to the surface?
                    pitch1 = (np.arctan2(b1[2], np.sqrt(b1[0]**2 + b1[1]**2)), 1)
                    pitch2 = (np.arctan2(b2[2], np.sqrt(b2[0]**2 + b2[1]**2)), 2)
                    pitch3 = (np.arctan2(b3[2], np.sqrt(b3[0]**2 + b3[1]**2)), 3)
                    min_pitch = min(pitch1, pitch2, pitch3)[1] # smallest angle (negative) is closest to surface and will be discarded
                    if min_pitch == 1:
                        ind_remove = ind_1
                        b1 = b3
                        ind_1 = ind_3
                    elif min_pitch == 2:
                        ind_remove = ind_2
                        b2 = b3
                        ind_2 = ind_3
                    else:
                        ind_remove = ind_3

                    unassigned_ind.remove(ind_remove)

                    if i+3 < nr_blobs:
                        dangle = abs(all_angles[sorted_indices[i+3]] - all_angles[sorted_indices[i]])
                        vert_dist = abs(all_blobs[2,sorted_indices[i+3]] - all_blobs[2,sorted_indices[i]]) #check that vertically aligned blobs have a certain min distance
                        if dangle < angle_thresh and vert_dist > vert_thresh: # a 4th blob from same direction?
                            ref = 2
                            ind_4 = sorted_indices[i+3]
                            b4 = all_blobs[:,ind_4]
                            # who is closest to the surface?
                            pitch1 = (np.arctan2(b1[2], sqrt(b1[0]**2 + b1[1]**2)) * 180 / pi, 1)
                            pitch2 = (np.arctan2(b2[2], sqrt(b2[0]**2 + b2[1]**2)) * 180 / pi, 2)
                            pitch4 = (np.arctan2(b4[2], sqrt(b4[0]**2 + b4[1]**2)) * 180 / pi, 4)
                            min_pitch = min(pitch1, pitch2, pitch4)[1] # smallest angle (negative)
                            if min_pitch == 1:
                                ind_remove = ind_1
                                b1 = b4
                                ind_1 = ind_4
                            elif min_pitch == 2:
                                ind_remove = ind_2
                                b2 = b4
                                ind_2 = ind_4
                            else:
                                ind_remove = ind_4

                            unassigned_ind.remove(ind_remove)

            # add final duo as neighbor
            if b2[2] < b1[2]:
                temp = b1
                b1 = b2
                b2 = temp

            pqr_twoblob = np.transpose(np.vstack((b1, b2)))
            xyz_twoblob = self._pqr_to_xyz(pqr_twoblob)
            xyz_twoblob_candidate.append(xyz_twoblob)
            twoblob_candidate_ind.append([ind_1, ind_2])

            i += 2 + ref
            neighbor_ind += 1
#       (return twoblob_candidate_ind, xyz_twoblob_candidate, unassigned_ind)
        #subfunction: match xyz_twoblob_candidate with predicted_blobs (input twoblob_candidate_ind, xyz_twoblob_candidate, unassigned_ind)
        if xyz_twoblob_candidate:
            xyz_led1_candidate = np.array([coord[:,0] for coord in xyz_twoblob_candidate])
            xyz_led2_candidate = np.array([coord[:,1] for coord in xyz_twoblob_candidate])

            xyz_led1_predicted = np.array([coord[:,0] for coord in predicted_blobs])
            xyz_led2_predicted = np.array([coord[:,1] for coord in predicted_blobs])

            led1_dist = cdist(xyz_led1_candidate, xyz_led1_predicted, 'cityblock')
            led2_dist = cdist(xyz_led2_candidate, xyz_led2_predicted, 'cityblock')
            dist_thresh = 500 #[mm]

            led1_dist = np.clip(led1_dist, 0, dist_thresh) #clip to threshold
            led2_dist = np.clip(led2_dist, 0, dist_thresh) #clip to threshold

            #add up normalized costs of both leds
            dist_normalized = (led1_dist + led2_dist)/(2*dist_thresh)
            #print("dist_normalized",dist_normalized)
            xyz_twoblob_matched_ind, track_twoblob_matched_ind = linear_sum_assignment(dist_normalized)
            xyz_twoblob_matched_ind = list(xyz_twoblob_matched_ind)
            track_twoblob_matched_ind = list(track_twoblob_matched_ind)

            #ignore matches with too high cost
            for i, j in zip(xyz_twoblob_matched_ind, track_twoblob_matched_ind):
                if dist_normalized[i,j] < 0.5:
                    unassigned_ind.difference_update(twoblob_candidate_ind[i]) #the blobs from those indices are assigned now
                else:
                    xyz_twoblob_matched_ind.remove(i)
                    track_twoblob_matched_ind.remove(j)

            #(return xyz_twoblob_matched_ind, track_twoblob_matched_ind, unassigned_ind, xyz_twoblob_new_ind)

            #subfunction: search 3rd led for matched vertical pairs (input: xyz_twoblob_matched_ind, track_twoblob_matched_ind, unassigned_ind, predicted_blobs, xyz_twoblob_candidate)
            vert_thresh = 0.2 #3rd led is allowed to be +- 20% of 86mm in height
            hor_thresh = 0.2 #3rd led is allowed to be +- 20% of 86mm in radial distance
            phi_thresh = pi/3 #[rad]

            xyz_twoblob_matched = [xyz_twoblob_candidate[i] for i in xyz_twoblob_matched_ind]
            xyz_twoblob_new_ind = list(set(range(neighbor_ind)).difference(set(xyz_twoblob_matched_ind)))

            i = 0
            while i < len(xyz_twoblob_matched):
                xyz_twoblob = xyz_twoblob_matched[i]
                phi_pred = predicted_phi[track_twoblob_matched_ind[i]]
                if not np.abs(xyz_twoblob[2,1]-xyz_twoblob[2,0]): #this shouldnt happen in real bots (blindspot), only because of rollover parsing
                    continue
                #find 3rd fitting led
                for j in unassigned_ind: #take all unassigned_ind indices
                    pqr_b3 = all_blobs[:,j]
                    xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                    xyz_b3 = xyz_threeblob[:,2]
                    phi = self._orientation(xyz_threeblob)
                    led_hor_dist = np.sqrt((xyz_b3[0]-xyz_twoblob[0,0])**2 + (xyz_b3[1]-xyz_twoblob[1,0])**2)
                    led_vert_dist = np.abs(xyz_b3[2]-xyz_twoblob[2,0])
                    #print(j, "predb3 to xyz_b3 norm", np.linalg.norm(xyz_b3_predicted - xyz_b3), "vert", led_vert_dist/U_LED_DZ, "hor", (np.abs(led_hor_dist-U_LED_DX)/U_LED_DX))
                    if led_vert_dist/U_LED_DZ < vert_thresh and (np.abs(led_hor_dist-U_LED_DX)/U_LED_DX) < hor_thresh and abs(np.arctan2(sin(phi-phi_pred), cos(phi-phi_pred))) < phi_thresh: #calculated led distances should be wihting range of real led distance DX, DZ
                        xyz_threeblob_matched.append(xyz_threeblob)
                        xyz_threeblob_matched_ind.append(xyz_twoblob_matched_ind[i])
                        track_threeblob_matched_ind.append(track_twoblob_matched_ind[i])
                        phi_matched.append(phi)
                        unassigned_ind.remove(j)
                        j_neighbor_ind = np.argwhere([j in x for x in twoblob_candidate_ind])
                        if any(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair --> remove that one
                            neighbor_ind -= 1
                            unassigned_ind.difference_update(twoblob_candidate_ind[j_neighbor_ind[0,0]])
                            try:
                                xyz_twoblob_new_ind.remove(j_neighbor_ind[0,0])
                            except ValueError:
                                pass # its not in the list
                        break
                    else:
                        continue
                i += 1

            #subfunction: second round to find 3rd led for xyz_twoblob_new: the once that havent matched with an existing track and want to start a new track (input)
            additional_twoblob_ind = list(set(xyz_twoblob_matched_ind).difference(xyz_threeblob_matched_ind))
            for i in additional_twoblob_ind:
                #print("additional", i)
                xyz_twoblob_new_ind.append(i)
                unassigned_ind.update(twoblob_candidate_ind[i])

            xyz_twoblob_new = [xyz_twoblob_candidate[i] for i in xyz_twoblob_new_ind]
            i = 0
            while i < len(xyz_twoblob_new):
                xyz_twoblob = xyz_twoblob_new[i]
                if not np.abs(xyz_twoblob[2,1]-xyz_twoblob[2,0]): #this shouldnt happen in real bots (blindspot), only because of rollover parsing
                    continue
                #find 3rd fitting led
                for j in [x for x in unassigned_ind if (x != twoblob_candidate_ind[xyz_twoblob_new_ind[i]][0] and x != twoblob_candidate_ind[xyz_twoblob_new_ind[i]][1])]:
                    pqr_b3 = all_blobs[:,j]
                    xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                    xyz_b3 = xyz_threeblob[:,2]
                    led_hor_dist = np.sqrt((xyz_b3[0]-xyz_twoblob[0,0])**2 + (xyz_b3[1]-xyz_twoblob[1,0])**2)
                    led_vert_dist = np.abs(xyz_b3[2]-xyz_twoblob[2,0])
                    if led_vert_dist/U_LED_DZ < vert_thresh and (np.abs(led_hor_dist-U_LED_DX)/U_LED_DX) < hor_thresh: #calculated led distances should be wihting range of real led distance DX, DZ
                        xyz_threeblob_new.append(xyz_threeblob)
                        xyz_threeblob_new_ind.append(xyz_twoblob_new_ind[i])
                        phi_new.append(self._orientation(xyz_threeblob))
                        unassigned_ind.remove(j)
                        unassigned_ind.difference_update(twoblob_candidate_ind[xyz_twoblob_new_ind[i]])
                        j_neighbor_ind = np.argwhere([j in x for x in twoblob_candidate_ind])
                        if any(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair --> remove that one
                            neighbor_ind -= 1
                            unassigned_ind.difference_update(twoblob_candidate_ind[j_neighbor_ind[0,0]])
                        break
                    else:
                        continue
                i += 1
        #print("matched nr", len(track_threeblob_matched_ind), "new nr", len(xyz_threeblob_new))
        return (xyz_threeblob_matched, phi_matched, xyz_threeblob_new, phi_new, track_threeblob_matched_ind)

    def home_orient(self, phi_des, v_des): #added pw
        """Homing behavior. Sets fin controls to move toward a desired goal orientation.

        Args:
            phi_des (np.array): Relative position of desired goal location in robot frame.
            v_des: desired linear velocity in phi direction
        """
        v_des = np.clip(v_des, -1, 1) #should already be in range -1,1
        F_P_max = 0.006 # [N] according to dynamics.py file
        F_caud_max = 0.020 # [N]
        angle_thresh = 2 # [deg]

        heading = phi_des*180/pi #pw stay directly in rad

        if abs(heading) < angle_thresh: #dont do anything if nearly aligned
            self.pect_r = 0
            self.pect_l = 0
        # target to the right
        elif heading > 0:
            self.pect_l = min(1, 0.6 + abs(heading) / 180) *0.25 #0.5 #pw added to reduce oscillation
            self.pect_r = 0
        # target to the left
        else:
            self.pect_r = min(1, 0.6 + abs(heading) / 180) *0.25 #0.5#pw added to reduce oscillation
            self.pect_l = 0

        self.caudal = np.clip(v_des + sin(self.dynamics.pect_angle)*(self.pect_r + self.pect_l)*F_P_max/F_caud_max, 0, 1) #to compensate for backwards movement of pectorial fins (inclided at 10deg)
        # move forward if aligned with swarm
        # if np.abs(heading) < np.pi/4:
        #     self.caudal = 0.7
        # else:
            # self.caudal = 0

    def move(self, detected_blobs):
        """Decision-making based on neighboring robots and corresponding move
        """
        self.it_counter += 1

        predicted_blobs, predicted_phi = self.kalman_prediction_update()
        # match blob triplets, give their orientation
        (xyz_matched, phi_matched, xyz_new, phi_new, matched_track_ind) = self.parse_orientation_reflections(detected_blobs, predicted_blobs, predicted_phi)

        self.kalman_measurement_update(xyz_matched, phi_matched, matched_track_ind) #predicted_ind has same length as xyz_matched and says to which track this measurement was matched

        self.kalman_remove_lost_tracks(matched_track_ind)

        (rel_pos_led1, rel_phi) = self.kalman_new_tracks(xyz_new, phi_new)

        # find target orientation
        center_orient = self.comp_center_orient(rel_phi)
        center_pos = self.comp_center_pos(rel_pos_led1)

        pred_alpha, pred_beta, predator_detected = self.environment.perceive_pred(self.id)

        if self.escaped:
            pred_alpha, pred_beta, pred_valpha, pred_vbeta = self.kalman_tracking_predator(pred_alpha, pred_beta)

        #print("pred_rel_phi, predator_detected",pred_rel_phi, predatpredator_detectedor_visible)

        if self.behavior == "aggregate":
            phi_des, v_des = self.aggregate(center_pos)

        if self.behavior == "align":
            phi_des, v_des = self.align(center_orient, pred_alpha, pred_beta, predator_detected)

        if self.behavior == "predator_escape":
            phi_des, v_des = self.predator_escape(center_pos, center_orient, pred_alpha, predator_detected)

        if self.behavior == "aggregate_and_turn":
            phi_des, v_des = self.aggregate_and_turn(center_pos)

        self.home_orient(phi_des, v_des)
        self.depth_ctrl_vert(center_pos[2])
        """ debugging
        self.dorsal = 0
        self.caudal = 0
        self.pect_r = 0
        self.pect_l = 0
        """
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        target_pos, self_vel = self.dynamics.simulate_move(self.id)

        return (target_pos, self_vel)
