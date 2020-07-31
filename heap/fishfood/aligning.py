"""Aligning
"""
from math import *
import numpy as np
from numpy import matlib
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

        #kf tracker init, move this to init
        self.kf_array = []
        self.first_detection = np.empty(0, dtype=int)
        self.last_detected = np.empty(0, dtype=int)
        self.track_id = np.empty(0, dtype=int)
        self.next_track_id = 0
        self.kf_phi_rotations = np.empty(0, dtype=int)
        self.kf_phi_prev = np.empty(0)

        self.wo_kf = False
        self.parsing_bool = self.environment.parsing_bool


        # Logger instance
        for filename in glob.glob("./logfiles/kf*"): #remove all previous files
            os.remove(filename)

        with open('./logfiles/kf_{}.csv'.format(self.id), 'w') as f:
            f.truncate()
            f.write('TRACK_ID, ITERATION, X, Y, Z, PHI, Avg Wrong parsed, Avg Correct parsed, parsing vector, 3rd led wrong \n')

    def init_kf(self, xyz_init, phi_init):
        dt = 0.5 #[s]#pw measure this!!!
        dim_state = 8
        dim_meas = 4 #only measure pos, not vel
        #dim_input = dim_state - dim_meas #only vel
        noise_process_xyz = 50 #*1000 #[mm] #this is only integration noise
        noise_process_phi = np.pi/8 #*1000#[rad] #this is only integration noise
        noise_process_vxyz = 100 #*1000#[mm/s] #speed prediction error
        noise_process_vphi = np.pi/4 #*1000 #[rad/s] #speed prediction error
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
                    '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                        self.track_id[i],
                        self.it_counter,
                        rel_pos[i][0],
                        rel_pos[i][1],
                        rel_pos[i][2],
                        rel_orient[i],
                        self.environment.parsing_wrong,
                        self.environment.parsing_correct,
                        self.environment.parsing_vector_track[i], #pw comment for wo
                        self.environment.parsing_wrong_third_led
                    )
                )

    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        if not self.parsing_bool:
            target_pos, vel = self.move_no_parsing(robots, rel_pos, duration)
        else:
            target_pos, vel = self.move(leds, duration)
        self.environment.update_states(self.id, target_pos, vel)

    def kalman_prediction_update(self):
        nr_tracks = len(self.kf_array)
        my_z = self.environment.pos[self.id, 2] #info from depth sensor

        #if there are no tracks, create a new one; pw should I?
        """
        if not nr_tracks:
            kf_new = self.init_kf(np.array([[0],[0],[0]]), np.array([0]))
            self.kf_array.append(kf_new)
            nr_tracks = 1
        """
        #prediction step
        for i in range(nr_tracks):
            self.kf_array[i].predict()

        #calc all led positions
        xyz_threeblob_all = []
        rel_orient = []
        for i in range(nr_tracks):
            predicted_state = self.kf_array[i].x
            xyz_led1 = predicted_state[0:3]
            xyz_led1[2] = max(-my_z, xyz_led1[2])#in case prediction is over water surface
            phi_bounded = np.arctan2(sin(predicted_state[3]), cos(predicted_state[3]))
            rel_orient.append(phi_bounded)
            xyz_threeblob = self.calc_all_led_xyz(xyz_led1, rel_orient[i])
            xyz_threeblob_all.append(xyz_threeblob)

        return xyz_threeblob_all, rel_orient

    def kalman_measurement_update(self, xyz, phi, track_ind):
        self.environment.parsing_vector_track = np.zeros((len(self.kf_array),1))
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
            self.environment.parsing_vector_track[i] = self.environment.parsing_vector[j]

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
        self.environment.parsing_vector_track = np.delete(self.environment.parsing_vector_track, delete_track_ind)

    def kalman_new_tracks(self, xyz_new, phi_new):
        nr_tracks = len(self.kf_array)
        # append new detections
        for j in range(len(phi_new)):
            #print("newtrack",xyz_new[j][:,0])
            self.kf_array.append(self.init_kf(xyz_new[j][:,0], phi_new[j]))

        #read out results
        rel_pos_led1 = []
        rel_phi = []
        if self.wo_kf: #each track will be only one element long
            nr_tracks = len(self.kf_array)

        for i in range(nr_tracks):#range(len(self.kf_array)):  the newest measurements will be ignored to prevent outliers
            estimated_state = self.kf_array[i].x
            xyz_led1 = estimated_state[0:3].ravel()
            if not np.array_equal(xyz_led1, np.array([0, 0, 0])): #empty init
                rel_pos_led1.append(xyz_led1)
                phi_bounded = np.arctan2(sin(estimated_state[3]), cos(estimated_state[3]))
                rel_phi.append(phi_bounded)
        #print("rel_pos_led1", rel_pos_led1)
        self.log_kf(rel_pos_led1, rel_phi)
        return (rel_pos_led1, rel_phi)

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
            self.dorsal = min(1, 0.7 + z_des/500) #pw add d-part to p controller? check useful values
        else:
            self.dorsal = 0.2 #pw what is value to hold depth?

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

        sqrt_pos = max(b**2 - 4 * a * c, 0) #pw preventing negative sqrt, if it is too far off, the blob will be discarded later by led_hor_dist
        d_plus = (-b + sqrt(sqrt_pos)) / (2 * a)
        d_minus = (-b - sqrt(sqrt_pos)) / (2 * a)

        diff_z_plus = abs(z1 - d_plus*pqr_3[2])
        diff_z_minus = abs(z1 - d_minus*pqr_3[2])
        #print("diff_z_plus,diff_z_minus",diff_z_plus,diff_z_minus)
        if (diff_z_plus < diff_z_minus): #choose solution for which led 3 is closer to same vertical height to led1
            xyz_3 = d_plus * pqr_3
        else:
            xyz_3 = d_minus * pqr_3

        xyz = np.append(xyz_1_2, xyz_3[:,np.newaxis], axis=1)

        return xyz

    def calc_relative_angles(self, blobs): #copied and adapted from BlueSwarm Code "avoid_duplicates_by_angle" #pw split this up in env and fish part?
        """Use right and left cameras just up to the xz-plane such that the overlapping camera range disappears and there are no duplicates.

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """
        angles = np.empty(0)
        for i in range(np.shape(blobs)[1]):
            led = blobs[:,i]
            angle = np.arctan2(led[1], led[0])
            angles = np.append(angles, angle)

        return angles #angles in rad!
    #""" new clean version
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
                print("impossible duplet")
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
            self.pect_l = min(1, 0.6 + abs(heading) / 180) *.25 #0.25 #pw added to reduce oscillation put back factor!!
            self.pect_r = 0
        # target to the left
        else:
            self.pect_r = min(1, 0.6 + abs(heading) / 180) *0.25 #0.25#pw added to reduce oscillation put back factor!!
            self.pect_l = 0

        self.caudal = np.clip(v_des + sin(self.dynamics.pect_angle)*(self.pect_r + self.pect_l)*F_P_max/F_caud_max, 0, 1) #to compensate for backwards movement of pectorial fins (inclided at 10deg)
        # move forward if aligned with swarm
        # if np.abs(heading) < np.pi/4:
        #     self.caudal = 0.7
        # else:
            # self.caudal = 0
    def home_aggregate_align(self, center_pos, center_orient):
        aggregatation_thresh = 500 #below this dont worry about aggregating
        aggregatation_max = 2000 #aggregate at full weight when at this distance
        center_heading = np.arctan2(center_pos[1], center_pos[0])
        center_dist = np.linalg.norm(center_pos[0:2])
        weight_agg = 0.75
        weight_agg *= center_dist/aggregatation_max #make weight_agg dependent on distance from center
        weight_align = 1 - weight_agg
        F_P_max = 0.006 # [N] according to dynamics.py file
        F_caud_max = 0.020 # [N]
        angle_thresh = 2 # [deg]
        v_des = 0 #just for init

        #print(self.id, center_dist)
        phi_des_align = center_orient

        if center_dist < aggregatation_thresh or center_pos[0] > 0:
            if center_dist < aggregatation_thresh: #dont do anything for aggregation if already close, just do aligning
                phi_des_agg = 0
                phi_des = phi_des_align
                v_des = 0

            elif center_pos[0] > 0: #center infront of me
                phi_des_agg = center_heading
                phi_des = weight_align * phi_des_align + weight_agg * phi_des_agg
                v_des = (1 - abs(phi_des) / pi)*weight_agg
                #v_des = np.interp(center_dist, (aggregatation_thresh, aggregatation_max), (0, 1)) # produce a v_des in range 0,1 depending on how far from center I'm away

            heading = phi_des*180/pi #pw stay directly in rad

            if abs(heading) < angle_thresh: #dont do anything if nearly aligned
                self.pect_r = 0
                self.pect_l = 0
                v_des = 0
            # target to the right
            elif heading > 0:
                self.pect_l = min(1, 0.6 + abs(heading) / 180) *.25 #0.25 #pw added to reduce oscillation put back factor!!
                self.pect_r = 0
            # target to the left
            else:
                self.pect_r = min(1, 0.6 + abs(heading) / 180) *.25 #0.25#pw added to reduce oscillation put back factor!!
                self.pect_l = 0
            self.caudal = np.clip(v_des + sin(self.dynamics.pect_angle)*(self.pect_r + self.pect_l)*F_P_max/F_caud_max, 0, 1) #to compensate for backwards movement of pectorial fins (inclided at 10deg)

        else:
            backwards_const = 0.5 * weight_agg

            phi_des_agg = np.arctan2(sin(pi-center_heading), cos(pi-center_heading))
            phi_des = weight_align * phi_des_align + weight_agg * phi_des_agg
            heading = phi_des*180/pi #pw stay directly in rad

            if abs(heading) < angle_thresh: #dont do anything if nearly aligned
                pass
            # target to the right
            elif heading < 0:
                self.pect_r = backwards_const
                self.pect_l = backwards_const + min(1-backwards_const, abs(heading) / 180) *1 #0.25 #pw added to reduce oscillation put back factor!!
            # target to the left
            else:
                self.pect_l = backwards_const
                self.pect_r = backwards_const + min(1-backwards_const, abs(heading) / 180) *1 #0.25#pw added to reduce oscillation put back factor!!

            if backwards_const:
                self.caudal = 0
            else:
                self.caudal = np.clip(sin(self.dynamics.pect_angle)*(self.pect_r + self.pect_l)*F_P_max/F_caud_max, 0, 1)
            #print(self.id, self.pect_l, self.pect_r)

    def move(self, detected_blobs, duration):
        """Decision-making based on neighboring robots and corresponding move
        """
        self.it_counter += 1

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

        # find target orientation
        center_orient = self.comp_center_orient(rel_phi)
        center_pos = self.comp_center_pos(rel_pos_led1)

        phi_des = center_orient
        #phi_des = (np.random.rand()-0.5)*pi #pw remove!!!
        """
        #to test spiral, pw:
        target = np.array([17800/2-self.environment.pos[self.id][0], 17800/2-self.environment.pos[self.id][1]])
        my_phi = self.environment.pos[self.id][3]
        target_rot = np.empty((2,1))
        target_rot[0] = cos(-my_phi)*target[0] - sin(-my_phi)*target[1]
        target_rot[1] = sin(-my_phi)*target[0] + cos(-my_phi)*target[1]
        pred_phi = np.arctan2(target_rot[1], target_rot[0]) #pw remove, fully towards center of tank-17800/2
        escape_angle = 60 *pi/180
        phi_des = np.arctan2(sin(pred_phi+escape_angle), cos(pred_phi+escape_angle))
        v_des = 0.05#0
        #print(self.id, phi_des)
        if np.linalg.norm(target) < 100:
            v_des = 0
        """
        v_des = 0
        self.home_orient(phi_des, v_des) #pw comment back!!
        #self.home_aggregate_align(center_pos, center_orient) #pw to test aggreagation
        #self.depth_ctrl_vert(center_pos[2])
        z_des = 300 + self.id *30 #so that they are in different heights #300+
        self.depth_ctrl_vert(z_des-self.environment.pos[self.id][2])
        """ debugging
        self.dorsal = 0
        self.caudal = 0
        self.pect_r = 0
        self.pect_l = 0
        """
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)

    def move_no_parsing(self, robots, rel_pos, duration):
        """Decision-making based on neighboring robots and corresponding move
        """
        self.it_counter += 1
        rel_pos_led1 = []
        rel_phi = []
        for i in range(np.shape(rel_pos)[0]):
            rel_pos_led1.append(rel_pos[i,:3])
            rel_phi.append(rel_pos[i,3])
        # find target orientation
        center_orient = self.comp_center_orient(rel_phi)
        center_pos = self.comp_center_pos(rel_pos_led1)

        phi_des = center_orient

        v_des = 0
        self.home_orient(phi_des, v_des)

        z_des = 500 + self.id *100 #so that they are in different heights #300 + self.id *30
        self.depth_ctrl_vert(z_des-self.environment.pos[self.id][2])
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)
