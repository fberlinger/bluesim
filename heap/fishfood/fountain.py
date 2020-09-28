"""Fountain maneuver
"""
from math import *
import numpy as np
import time
import os

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
        self.escape_started = False
        self.pred_started = False
        self.pred_undetected_count = 0
        self.escape_angle = self.environment.escape_angle
        self.dt = 1/self.environment.clock_freq

        #kf tracker init, move this to init
        self.kf_array = []
        self.first_detection = np.empty(0, dtype=int)
        self.last_detected = np.empty(0, dtype=int)
        self.track_id = np.empty(0, dtype=int)
        self.next_track_id = 0
        self.kf_phi_rotations = np.empty(0, dtype=int)
        self.kf_phi_prev = np.empty(0)

        self.wo_kf = False
        self.through_blindspot = False
        self.phi_des_history = []
        self.turning_counter = 0

        # Logger instance
        with open('./logfiles/kf_{}.csv'.format(self.id), 'w') as f:
            f.truncate()
            f.write('TRACK_ID, ITERATION, X, Y, Z, PHI \n')

    def init_kf(self, xyz_init, phi_init):
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
        kf.F = np.identity(dim_state) + np.pad(self.dt*np.identity(dim_meas), ((0, 4), (4, 0)), 'constant',constant_values = (0,0)) #transition matrix: assume const vel; PW in BlueSwarm code should F be changed every iteration because dt is not const? Time-varying kalman filter
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

    def init_kf_pred(self, pred_xyzphi_init):
        dim_state = 8 #state contains: x, y, z, phi, vx, vy, vz, vphi
        dim_meas = 4 #only measure pos, not vel
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
        kf.x = np.concatenate((pred_xyzphi_init, v_init), axis=None)[:,np.newaxis]
        kf.F = np.identity(dim_state) + np.pad(self.dt*np.identity(dim_meas), ((0, 4), (4, 0)), 'constant',constant_values = (0,0)) #transition matrix: assume const vel; PW in BlueSwarm code should F be changed every iteration because dt is not const? Time-varying kalman filter
        kf.H = np.pad(np.identity(dim_meas), ((0, 0), (0, 4)), 'constant',constant_values = (0,0)) #measurement matrix: we measure pos
        #kf.B = np.append(np.zeros((4,4)), np.identity(dim_input), axis=0) #control matrix: u is directy imu input vel (integrated acc) and v_phi (gyro), so B is zero on position and identity on velocity
        kf.R = np.diag([noise_meas_xyz, noise_meas_xyz, noise_meas_xyz, noise_meas_phi]) #measurement noise
        kf.Q = np.diag([noise_process_xyz, noise_process_xyz, noise_process_xyz, noise_process_phi, noise_process_vxyz, noise_process_vxyz, noise_process_vxyz, noise_process_vphi]) #process noise #Q_discrete_white_noise(dim=dim_state, dt=dt, var=var_state)
        kf.P = np.diag([covar_init_xyz, covar_init_xyz, covar_init_xyz, covar_init_phi, covar_init_vxyz, covar_init_vxyz, covar_init_vxyz, covar_init_vphi]) #estimated initial covariance matrix (a measure of the estimated accuracy of the state estimate)

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

    def log_kf(self, rel_pos, rel_orient, pred_pos):
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
            #predator
            if pred_pos.size:
                f.write(
                    '{}, {}, {}, {}, {}, {}\n'.format(
                        1000,
                        self.it_counter,
                        pred_pos[0],
                        pred_pos[1],
                        pred_pos[2],
                        pred_pos[3]
                    )
                )

    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        pred_rel_pos = self.environment.perceive_pred(self.id, robots, rel_pos, dist)
        target_pos, vel = self.move(leds, pred_rel_pos, duration)
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
        return (rel_pos_led1, rel_phi)

    def kalman_tracking_predator(self, pred_pos_detection):
        #prediction step
        self.kf_pred.predict()

        #measurements
        if pred_pos_detection.size:
            self.kf_pred.update(pred_pos_detection)

        return self.kf_pred.x[:4].ravel() #(xyz, phi)

    def kalman_update_predator(self, pred_pos):
        if pred_pos.size:
            self.pred_undetected_count = 0
            if not self.pred_started:
                # init predator kf
                self.kf_pred = self.init_kf_pred(pred_pos)
                self.pred_started = True

        elif not pred_pos.size and self.pred_started:
            self.pred_undetected_count += 1
            if self.pred_undetected_count > 20:
                self.pred_started = False
                self.behavior = "aggregate"
                print(self.id, "aggregate cause pred not seen")

        if self.pred_started:
            pred_pos = self.kalman_tracking_predator(pred_pos)

        return pred_pos

    def aggregate(self, center_pos):
        phi_des = np.arctan2(center_pos[1], center_pos[0])
        v_des = (1 - abs(phi_des)/pi)#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
        dist_thresh = 200 #mm
        dist_center = center_pos[0]**2 + center_pos[1]**2
        #print(self.id, "dist_center",dist_center)
        if dist_center < dist_thresh**2:
            print(self.id, "aligning")
            self.behavior = "align"
        return (phi_des, v_des)

    def align(self, center_orient, pred_pos):
        v_des = 0
        phi_des = center_orient
        if pred_pos.size and abs(pred_pos[3])<pi/2 and pred_pos[0]<0 and np.linalg.norm(pred_pos[0:2]) < 2000 and not self.escape_started: #if predator is visible and behind me and I havent already done a previous escape maneuver
            self.behavior = "predator_escape"
            print(self.id, "predator_escape")
            self.escape_started = True
            pred_alpha = np.arctan2(pred_pos[1], pred_pos[0])
            angle_sum =  pred_alpha - center_orient #correct my own perspective by my orientation towards swarm pw problem is that swarm is already swimming away --> center_orient not meaningful
            if np.arctan2(sin(angle_sum), cos(angle_sum)) < 0:
            #if pred_pos[1] < 0:
                self.escape_angle *= -1

        return (phi_des, v_des)

    def predator_escape(self, center_pos, center_orient, pred_pos):
        history_len_max = 35
        history_len_min = 10
        history_len_win = []
        #check if through_blindspot
        if not self.through_blindspot:
            if len(pred_pos) and (np.sign(self.escape_angle) * np.sign(pred_pos[1])) == -1:
                self.through_blindspot = True
                print(self.id, "through_blindspot")

        if len(pred_pos):
            pred_angle = np.arctan2(pred_pos[1], pred_pos[0])
            angle_sum = pred_angle + self.escape_angle
            phi_des = np.arctan2(sin(angle_sum), cos(angle_sum))
            self.phi_des_history.append(phi_des)
            if len(self.phi_des_history) > history_len_max:
                self.phi_des_history.pop(0)
            v_des = min(1, (2 - 2*abs(phi_des)/pi))#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
            if np.linalg.norm(pred_pos[:2]) > 2500: #pw tune max distance
                self.behavior = "keep_turning"
                print(self.id, "pred too far away, keep turning")

        else: #if no pred detected, take prev phi
            phi_des = 0
            v_des = 0
            self.phi_des_history.append(phi_des)
            if len(self.phi_des_history) > history_len_max:
                self.phi_des_history.pop(0)

            if not self.through_blindspot: #or found_flash : #if I didnt see the pred but the others are still blinking, keep doing the same thing as before for a longer interval than if noone is blinking
                history_len_win = history_len_max
            else:
                history_len_win = history_len_min
            for phi_last in list(reversed(self.phi_des_history))[:history_len_win]:
                if phi_last:
                    phi_des = phi_last
                    v_des = min(1, (2 - 2*abs(phi_des)/pi))#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?
                    break

            if not phi_des:
                self.behavior = "keep_turning"
                print(self.id, "keep turning")


            # v_des = min(1, (2 - 2*abs(phi_des)/pi))#make v_des dependend on phi_des : if aligned swim faster, pw choose non linear relation?

            # dist_thresh = 500 #mm
            # dist_center = center_pos[0]**2 + center_pos[1]**2
            #print("dist_center",dist_center)

            # if dist_center and dist_center < dist_thresh**2 and abs(pred_pos[3]) > pi*1/3 and pred_pos[3]*pred_alpha > 0:#and abs(pred_phi) < pi*2/3 #same sign alpha and phi #if Im close to my friends and i've been escaping already for some time for normal tank size: 60, find better rule!!
            #     print(self.id, "aggregate_and_turn cause at dist:", sqrt(dist_center), "pred phi", pred_pos[3])
            #     self.behavior = "aggregate_and_turn"

        return (phi_des, v_des)

    def keep_turning(self):
        phi_des = -np.sign(self.escape_angle)*90/180*pi
        v_des = 0
        self.turning_counter += 1
        if self.turning_counter > 20:
            self.behavior = "align"
            print(self.id, "align")
        return (phi_des, v_des)

    def aggregate_and_turn(self, center_pos, center_orient, pred_pos): #if pred is still visible, but so far away that we dont need to be scared anymore
        phi_aggregating = np.arctan2(center_pos[1], center_pos[0])

        if not pred_pos.size:
            phi_turning = -np.sign(self.escape_angle)*90/180*pi
        else:
            phi_turning = pred_pos[3]
        a = 0.3 #weight aggregation; pw tune
        b = 1 - a #weight turning
        phi_des = a*phi_aggregating + b*phi_turning

        dist_thresh = 50 #mm
        dist_center = center_pos[0]**2 + center_pos[1]**2
        #print("dist_center",dist_center)
        if dist_center < dist_thresh**2:
            print(self.id, "aligning")
            self.behavior = "align"

        v_des = (1 - abs(phi_des)/pi)
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

    def remove_reflections(self, unassigned_ind, my_z, duplet, inthresh_pitch_sorted, inthresh_ind_sorted):
        pitch_thresh =  1 / 180*np.pi #pw tune, before 1deg
        z_ref_pred1 = - (2*my_z + duplet[2,0]) #predicted z coordinate of reflection
        z_ref_pred2 = - (2*my_z + duplet[2,1])
        pitch_ref_pred1 = np.arctan2(z_ref_pred1, sqrt(duplet[0,0]**2 + duplet[1,0]**2))
        pitch_ref_pred2 = np.arctan2(z_ref_pred2, sqrt(duplet[0,1]**2 + duplet[1,1]**2))

        pitch_diff1 = abs(inthresh_pitch_sorted - pitch_ref_pred1)
        pitch_diff2 = abs(inthresh_pitch_sorted - pitch_ref_pred2)
        ind_ref = np.argwhere(np.logical_or(pitch_diff1 < pitch_thresh, pitch_diff2 < pitch_thresh)).ravel()
        if len(ind_ref):
            #print("removed {} refl, at angle".format(len(ind_ref)), np.array(all_angles)[inthresh_ind_sorted[ind_ref]])
            unassigned_ind.difference_update(inthresh_ind_sorted[ind_ref]) #remove all ind which are close to predicted reflection
            wrong_refl_removal = self.environment.count_wrong_refl_removal(inthresh_ind_sorted[ind_ref])
        return unassigned_ind

    def parsing(self, detected_blobs, predicted_blobs, predicted_phi):
        nr_blobs = np.shape(detected_blobs)[1]

        if nr_blobs < 3: # 0 robots
            return ([], [], [], [], [])
        my_z = self.environment.pos[self.id, 2] #info from depth sensor

        duplet_candidates, duplet_candidates_ind, unassigned_ind = self.find_duplet_candidates(detected_blobs, my_z)
        if not duplet_candidates:
            return ([], [], [], [], [])
        #match
        duplet_matched_ind, track_matched_duplet_ind, duplet_new_ind, unassigned_ind = self.match_duplet_tracks(duplet_candidates, duplet_candidates_ind, unassigned_ind, predicted_blobs)
        triplet_matched, phi_matched, triplet_matched_ind, track_matched_ind, duplet_new_ind, b3_matched_ind, unassigned_ind = self.find_triplet_matched(detected_blobs, duplet_candidates, duplet_candidates_ind, duplet_matched_ind, track_matched_duplet_ind, unassigned_ind, predicted_blobs, predicted_phi) #triplet_matched is xyz of led1 and phi
        #new track
        triplet_new, phi_new = self.find_triplet_new(detected_blobs, duplet_candidates, duplet_candidates_ind, duplet_new_ind, unassigned_ind)
        #this function evaluates the percentage of correctly matched leds

        self.environment.count_wrong_parsing(duplet_candidates_ind, duplet_matched_ind, b3_matched_ind, detected_blobs, self.id)

        return (triplet_matched, phi_matched, triplet_new, phi_new, track_matched_ind)

    def find_duplet_candidates(self, detected_blobs, my_z):
        duplet_candidates = []
        duplet_candidates_ind = []
        unassigned_ind = []
        nr_blobs = np.shape(detected_blobs)[1]

        detected_blobs_angles = self.calc_relative_angles(detected_blobs)
        sorted_ind = np.argsort(detected_blobs_angles)
        unassigned_ind = set(range(nr_blobs))
        angle_thresh =  2 / 180*np.pi #0.000001 #pw tune, before 2
        no_duplets = 0
        i = 0
        while i+1 < nr_blobs: # iterate through all blobs and fill array
            inthresh_len = sum(detected_blobs_angles[sorted_ind[i:]] < detected_blobs_angles[sorted_ind[i]] + angle_thresh)
            if inthresh_len < 2: #cant be a duplet with only one blob
                i += 1
                continue

            inthresh_ind = sorted_ind[i:i+inthresh_len]
            inthresh_pitch = []
            #calculate pitch to find the two lowest leds
            for ind in inthresh_ind:
                pitch = np.arctan2(detected_blobs[2, ind], sqrt(detected_blobs[0, ind]**2 + detected_blobs[1, ind]**2))
                inthresh_pitch.append(pitch)
            pitch_sort_ind = np.argsort(inthresh_pitch)[::-1] #backwards so that is starts at largest pitch = not reflection, but inside water
            inthresh_pitch_sorted = np.array(inthresh_pitch)[pitch_sort_ind]
            inthresh_ind_sorted = np.array(inthresh_ind)[pitch_sort_ind]
            inthresh_blobs_sorted = np.array(detected_blobs)[:, inthresh_ind_sorted]

            pqr_duplet = np.transpose(np.vstack((inthresh_blobs_sorted[:, 1], inthresh_blobs_sorted[:, 0]))) # LED1: second lowest pitch, LED2: lowest pitch
            duplet = self._pqr_to_xyz(pqr_duplet)

            if my_z + duplet[2,0] < 0 or np.linalg.norm(duplet[:,0]) > 4000 or np.linalg.norm(duplet[:,0]) < 50: #LED 1 is above water surface or fish too close or too far away --> impossible, continue
                i += 1
                #print("impossible duplet")
                continue

            duplet_candidates.append(duplet)
            duplet_candidates_ind.append([inthresh_ind_sorted[1], inthresh_ind_sorted[0]]) #take those two with lowest pitch
            no_duplets += 1
            i += inthresh_len

            unassigned_ind = self.remove_reflections(unassigned_ind, my_z, duplet, inthresh_pitch_sorted, inthresh_ind_sorted)

        return duplet_candidates, duplet_candidates_ind, unassigned_ind

    def match_duplet_tracks(self, duplet_candidates, duplet_candidates_ind, unassigned_ind, predicted_blobs):

        duplet_matched_ind = []
        track_matched_duplet_ind = []
        duplet_new_ind = []
        if not predicted_blobs:
            return duplet_matched_ind, track_matched_duplet_ind, duplet_new_ind, unassigned_ind

        led1_candidate = np.array([coord[:,0] for coord in duplet_candidates])
        led2_candidate = np.array([coord[:,1] for coord in duplet_candidates])

        led1_predicted = np.array([coord[:,0] for coord in predicted_blobs])
        led2_predicted = np.array([coord[:,1] for coord in predicted_blobs])

        led1_dist = cdist(led1_candidate, led1_predicted, 'euclidean')
        led2_dist = cdist(led2_candidate, led2_predicted, 'euclidean')
        D = (led1_dist + led2_dist)/2
        # dist_thresh =  200 + 100*np.matlib.repmat(self.first_detection, np.shape(D)[0], 1) #300 #mm, pw tune
        dist_thresh = 300 #mm, pw tune
        #"""#greedy matching (pw: is better than munkres!)
        while np.min(D) < dist_thresh:
            ind = np.unravel_index(np.argmin(D, axis=None), D.shape)
            D[ind[0],:] = dist_thresh + 1
            D[:,ind[1]] = dist_thresh + 1
            duplet_matched_ind.append(ind[0])
            track_matched_duplet_ind.append(ind[1])
            unassigned_ind.difference_update(duplet_candidates_ind[ind[0]]) #those blobs are assigned now

        """#Munkres
        #dist_thresh = 400 + 200*np.matlib.repmat(first_detection, np.shape(led1_dist)[0], 1) #[mm] larger thresh if first detection
        duplet_matched_ind_long, track_matched_duplet_ind_long = linear_sum_assignment(D)

        #sort out matches with too high cost
        for i, j in zip(list(duplet_matched_ind_long), list(track_matched_duplet_ind_long)):
            if D[i,j] < dist_thresh: #cost_thresh for munkres
                duplet_matched_ind.append(i)
                track_matched_duplet_ind.append(j)
                unassigned_ind.difference_update(duplet_candidates_ind[i]) #the blobs from those indices are assigned now -> remove them from unassigned_ind

        #"""
        #print(duplet_matched_ind, track_matched_duplet_ind, duplet_new_ind, unassigned_ind)
        return duplet_matched_ind, track_matched_duplet_ind, duplet_new_ind, unassigned_ind

    def find_triplet_matched(self, detected_blobs, duplet_candidates, duplet_candidates_ind, duplet_matched_ind, track_matched_duplet_ind, unassigned_ind, predicted_blobs, predicted_phi):
        triplet_matched = []
        triplet_matched_ind = []
        track_matched_ind = []
        phi_matched = []
        b3_matched_ind = []
        duplet_new_ind = set(range(len(duplet_candidates)))
        if not (duplet_matched_ind and unassigned_ind and predicted_blobs):
            return triplet_matched, phi_matched, triplet_matched_ind, track_matched_ind, duplet_new_ind, b3_matched_ind, unassigned_ind

        unassigned_ind_list = list(unassigned_ind)
        N0 = len(duplet_matched_ind)
        N1 = len(unassigned_ind_list)
        D = np.empty((N0, N1))
        phi_thresh = pi/3 #rad
        xyz_thresh = 10#20 #mm
        #fill in cost matrix
        # for idx_i, i in enumerate(duplet_matched_ind):
        #     duplet = duplet_candidates[i]
        #     b3_predicted = predicted_blobs[track_matched_duplet_ind[idx_i]][:,2]
        #     for idx_j, j in enumerate(unassigned_ind_list): #take all unassigned_ind indices
        #         b3_pqr = detected_blobs[:,j]
        #         triplet = self._pqr_3_to_xyz(duplet, b3_pqr)
        #         b3 = triplet[:,2]
        #         dist = np.linalg.norm(b3 - b3_predicted)
        #         D[idx_i, idx_j] = dist
        #         #print(j, "predb3 to xyz_b3 norm", np.linalg.norm(xyz_b3_predicted - xyz_b3), "vert", led_vert_dist/U_LED_DZ, "hor", (np.abs(led_hor_dist-U_LED_DX)/U_LED_DX))
        # dist_thresh = 60 #mm, pw tune, euclidean dist from predicted led3 to measured blob
        for idx_i, i in enumerate(duplet_matched_ind):
            duplet = duplet_candidates[i]
            b3_predicted = predicted_blobs[track_matched_duplet_ind[idx_i]][:,2]
            phi_predicted = predicted_phi[track_matched_duplet_ind[idx_i]]
            for idx_j, j in enumerate(unassigned_ind_list): #take all unassigned_ind indices
                b3_pqr = detected_blobs[:,j]
                triplet = self._pqr_3_to_xyz(duplet, b3_pqr)
                b3 = triplet[:,2]
                phi = self._orientation(triplet)
                hor_dist = abs(sqrt((b3[0]-duplet[0,0])**2 + (b3[1]-duplet[1,0])**2) - U_LED_DX) #radial deviation from circle
                vert_dist = abs(b3[2]-duplet[2,0]) #vertical deviation from circle
                phi_dist = abs(np.arctan2(sin(phi-phi_predicted), cos(phi-phi_predicted)))
                circle_dist = sqrt(hor_dist**2 + vert_dist**2)
                dist = (phi_dist/phi_thresh + circle_dist/xyz_thresh)/2
                D[idx_i, idx_j] = dist
                #print("hor_dist",hor_dist, "vert_dist", vert_dist,"phi_dist",phi_dist )
        dist_thresh = 1 # normalized cost
        #"""#greedy matching (pw: is better than munkres!)
        try:
            while np.min(D) < dist_thresh:
                ind = np.unravel_index(np.argmin(D, axis=None), D.shape)
                D[ind[0],:] = dist_thresh + 1
                D[:,ind[1]] = dist_thresh + 1
                i = duplet_matched_ind[ind[0]]
                j = unassigned_ind_list[ind[1]]
                #calc triplet
                duplet = duplet_candidates[i]
                b3_pqr = detected_blobs[:,j]
                triplet = self._pqr_3_to_xyz(duplet, b3_pqr)
                phi = self._orientation(triplet)
                #append triplet
                triplet_matched.append(triplet)
                triplet_matched_ind.append(duplet_matched_ind[ind[0]])
                track_matched_ind.append(track_matched_duplet_ind[ind[0]])
                phi_matched.append(phi)
                b3_matched_ind.append([duplet_candidates_ind[i][0],duplet_candidates_ind[i][1],j])
                unassigned_ind.discard(j)
                j_neighbor_ind = np.argwhere([j in x for x in duplet_candidates_ind])
                if np.size(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair
                    unassigned_ind.difference_update(duplet_candidates_ind[j_neighbor_ind[0,0]]) # remove the blob that has been paired with j, cause it's probably its reflection
                    duplet_new_ind.discard(j_neighbor_ind[0,0])

            duplet_new_ind.difference_update(triplet_matched_ind)
        except:
            print("invalid cost mat",D)

        """#Munkres

        duplet_matched_idx, thirdblob_matched_idx = linear_sum_assignment(D)

        #check if cost of matched 3rd led is smaller than thresh
        for idx_i, idx_j in zip(duplet_matched_idx, thirdblob_matched_idx):
            if D[idx_i, idx_j] < dist_thresh:
                i = duplet_matched_ind[idx_i]
                j = unassigned_ind_list[idx_j]
                duplet = duplet_candidates[i]
                b3_pqr = detected_blobs[:,j]
                triplet = self._pqr_3_to_xyz(duplet, b3_pqr)
                phi = self._orientation(triplet)
                #append matched triplet
                triplet_matched.append(triplet)
                b3_matched_ind.append([duplet_candidates_ind[i][0], duplet_candidates_ind[i][1], j])
                triplet_matched_ind.append(i)
                track_matched_ind.append(track_matched_duplet_ind[idx_i])
                phi_matched.append(phi)
                unassigned_ind.discard(j)
                j_neighbor_ind = np.argwhere([j in x for x in duplet_candidates_ind])
                if np.size(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair
                    unassigned_ind.difference_update(duplet_candidates_ind[j_neighbor_ind[0,0]]) # remove the blob that has been paired with j, cause it's probably its reflection
                    duplet_new_ind.discard(j_neighbor_ind[0,0])
        #"""

        return triplet_matched, phi_matched, triplet_matched_ind, track_matched_ind, duplet_new_ind, b3_matched_ind, unassigned_ind

    def find_triplet_new(self, detected_blobs, duplet_candidates, duplet_candidates_ind, duplet_new_ind, unassigned_ind):
        triplet_new = []
        phi_new = []
        duplet_new_ind_list = list(duplet_new_ind)
        unassigned_ind_list = list(unassigned_ind)

        dist_thresh = 10 #mm, pw tune, dist thresh of matched led3 to circle
        for i in duplet_new_ind_list:
            if i not in duplet_new_ind:
                continue
            duplet = duplet_candidates[i]
            #find 3rd fitting led
            for j in unassigned_ind_list:
                if j not in unassigned_ind or j in duplet_candidates_ind[i]: #this j is already used to form the duplet
                    continue
                b3_pqr = detected_blobs[:,j]
                triplet = self._pqr_3_to_xyz(duplet, b3_pqr)
                b3 = triplet[:,2]
                hor_dist = abs(sqrt((b3[0]-duplet[0,0])**2 + (b3[1]-duplet[1,0])**2) - U_LED_DX) #radial deviation from circle
                vert_dist = abs(b3[2]-duplet[2,0]) #vertical deviation from circle
                dist = sqrt(hor_dist**2 + vert_dist**2)
                if dist < dist_thresh:
                    phi = self._orientation(triplet)
                    triplet_new.append(triplet)
                    phi_new.append(phi)
                    unassigned_ind.discard(j)
                    unassigned_ind.difference_update(duplet_candidates_ind[i])
                    j_neighbor_ind = np.argwhere([j in x for x in duplet_candidates_ind])
                    if np.size(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair --> remove that one and the vertical blobs from the unassigned list
                        unassigned_ind.difference_update(duplet_candidates_ind[j_neighbor_ind[0,0]])
                        duplet_new_ind.discard(j_neighbor_ind[0,0])

                    break

        return triplet_new, phi_new

    def track_neighbors(self, detected_blobs):
        if self.wo_kf:
            predicted_blobs = []
            predicted_phi = []
        else:
            predicted_blobs, predicted_phi = self.kalman_prediction_update()
        # match blob triplets, give their orientation
        (xyz_matched, phi_matched, xyz_new, phi_new, matched_track_ind) = self.parsing(detected_blobs, predicted_blobs, predicted_phi)

        self.kalman_measurement_update(xyz_matched, phi_matched, matched_track_ind) #predicted_ind has same length as xyz_matched and says to which track this measurement was matched

        self.kalman_remove_lost_tracks(matched_track_ind)

        (rel_pos_led1, rel_phi) = self.kalman_new_tracks(xyz_new, phi_new)

        return rel_pos_led1, rel_phi

    def home_orient(self, phi_des, v_des): #added pw
        """Homing behavior. Sets fin controls to move toward a desired goal orientation.

        Args:
            phi_des (np.array): Relative position of desired goal location in robot frame.
            v_des: desired linear velocity in phi direction
        """
        v_des *= self.environment.fish_factor_speed #slow down by this factor (before the v_des sould already be in range -1,1)
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

    def move(self, detected_blobs, pred_pos, duration):
        """Decision-making based on neighboring robots and corresponding move
        """
        self.it_counter += 1

        (rel_pos_led1, rel_phi) = self.track_neighbors(detected_blobs)

        # find target orientation
        center_orient = self.comp_center_orient(rel_phi)
        center_pos = self.comp_center_pos(rel_pos_led1)

        #pred_pos = self.kalman_update_predator(pred_pos)

        self.log_kf(rel_pos_led1, rel_phi, pred_pos)

        if self.behavior == "aggregate":
            phi_des, v_des = self.aggregate(center_pos)

        if self.behavior == "align":
            phi_des, v_des = self.align(center_orient, pred_pos)

        if self.behavior == "predator_escape":
            phi_des, v_des = self.predator_escape(center_pos, center_orient, pred_pos)

        # if self.behavior == "aggregate_and_turn":
        #     phi_des, v_des = self.aggregate_and_turn(center_pos, center_orient, pred_pos)
        if self.behavior == "keep_turning":
            phi_des, v_des = self.keep_turning()

        self.home_orient(phi_des, v_des)
        #self.depth_ctrl_vert(center_pos[2])
        z_des = 1000 # pw as if we had a pressure sensor
        self.depth_ctrl_vert(z_des-self.environment.pos[self.id][2])
        """ debugging
        self.dorsal = 0
        self.caudal = 1 #why is predator slower than fish
        self.pect_r = 0
        self.pect_l = 0
        """
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l, self.id)
        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)