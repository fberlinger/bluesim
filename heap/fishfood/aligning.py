"""Aligning
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
            f.write('TRACK_ID, ITERATION, X, Y, Z, PHI, Wrong parsed, Correct parsed \n')

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
                    '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                        self.track_id[i],
                        self.it_counter,
                        rel_pos[i][0],
                        rel_pos[i][1],
                        rel_pos[i][2],
                        rel_orient[i],
                        self.environment.parsing_wrong,
                        self.environment.parsing_correct
                    )
                )

    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        target_pos, vel = self.move(leds, duration)
        self.environment.update_states(self.id, target_pos, vel)

    def kalman_prediction_update(self):
        nr_tracks = len(self.kf_array)

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
        xyz_twoblob_matched_ind = []
        xyz_threeblob_matched = []
        xyz_threeblob_matched_ind = []
        track_threeblob_matched_ind = []
        phi_matched = []
        phi_new = []
        xyz_threeblob_new = []
        xyz_threeblob_new_ind = []
        blob3_matched_ind = []


        nr_blobs = np.shape(all_blobs)[1]
        if nr_blobs < 3: # 0 robots
            return (xyz_threeblob_matched, phi_matched, xyz_threeblob_new, phi_new, track_threeblob_matched_ind)

        # find all valid blobs and their respective angles
        all_angles = self.calc_relative_angles(all_blobs)

        # subfunction: find vertically aligned leds (xyz_twoblob_candidates) and sort out reflections where >2 blobs have similar angle
        sorted_indices = np.argsort(all_angles)
        unassigned_ind = set(range(nr_blobs))
        angle_thresh = 2 / 180*np.pi#0.000001 / 180*np.pi #  only right now to check kf performance really really small, works only in noiseless simulation # below which 2 blobs are considered a duo (3deg) or smaller: only take out the obvious reflections, leave the others to the kf
        #vert_thresh = 0.0001 #pqr normalized
        pitch_thresh = 0 / 180*np.pi # if the blobs are not far enough apart in vertical direction, assume its not a reflection but maybe the third led

        i = 0 # blob_ind
        neighbor_ind = 0
        while i+1 < nr_blobs: # iterate through all blobs and fill array
            ind_1 = sorted_indices[i]
            ind_2 = sorted_indices[i+1]
            b1 = all_blobs[:,ind_1]
            b2 = all_blobs[:,ind_2]
            # if 2 blobs are too far apart, ignore first one and check next 2
            dangle = abs(all_angles[ind_1] - all_angles[ind_2])
            #vert_dist = abs(b2[2] - b1[2]) #check that vertically aligned blobs have a certain min distance
            #print("vert_dist",vert_dist)
            pitch1 = np.arctan2(b1[2], np.sqrt(b1[0]**2 + b1[1]**2))
            pitch2 = np.arctan2(b2[2], np.sqrt(b2[0]**2 + b2[1]**2))
            if not(dangle < angle_thresh and abs(pitch1 - pitch2) > pitch_thresh):#vert_dist > vert_thresh):
                # if dangle < angle_thresh and vert_dist < vert_thresh: #assume blobs are identical if the angles and vert distance is close together --> discard one of them
                #     unassigned_ind.remove(ind_1) # pw is this necessary?? why??
                i += 1
                continue

            # else, continue and add 2 blobs

            # check for reflections
            ref = 0

            if i+2 < nr_blobs:
                ind_3 = sorted_indices[i+2]
                b3 = all_blobs[:,ind_3]
                dangle = abs(all_angles[ind_3] - all_angles[ind_1])
                #vert_dist = abs(all_blobs[2,sorted_indices[i+2]] - all_blobs[2,sorted_indices[i]]) #check that vertically aligned blobs have a certain min distance
                pitch3 = np.arctan2(b3[2], np.sqrt(b3[0]**2 + b3[1]**2))
                if dangle < angle_thresh and min(abs(pitch1 - pitch3), abs(pitch2 - pitch3)) > pitch_thresh: #vert_dist > vert_thresh: # a 3rd blob from same direction?
                    #print("surfref 1, ",min(abs(pitch1 - pitch3), abs(pitch2 - pitch3)))
                    ref = 1
                    # who is closest to the surface?
                    min_pitch = np.argmin([pitch1, pitch2, pitch3]) + 1 # smallest angle (negative) is closest to surface and will be discarded
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
                        ind_4 = sorted_indices[i+3]
                        b4 = all_blobs[:,ind_4]
                        dangle = abs(all_angles[ind_4] - min(all_angles[ind_1], all_angles[ind_2]))
                        #vert_dist = abs(all_blobs[2,sorted_indices[i+3]] - all_blobs[2,sorted_indices[i]]) #check that vertically aligned blobs have a certain min distance
                        pitch1 = np.arctan2(b1[2], sqrt(b1[0]**2 + b1[1]**2))
                        pitch2 = np.arctan2(b2[2], sqrt(b2[0]**2 + b2[1]**2))
                        pitch4 = np.arctan2(b4[2], sqrt(b4[0]**2 + b4[1]**2))
                        if dangle < angle_thresh and min(abs(pitch1 - pitch4), abs(pitch2 - pitch4)) > pitch_thresh:#vert_dist > vert_thresh: # a 4th blob from same direction?
                            #print("surfref 2, ",min(abs(pitch1 - pitch4), abs(pitch2 - pitch4)))
                            ref = 2
                            # who is closest to the surface?
                            min_pitch = np.argmin([pitch1, pitch2, pitch4]) + 1  # smallest angle (negative)
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
            if b2[2] < b1[2]: #swap so that blob 1 is closer to surface
                temp = b1
                b1 = b2
                b2 = temp
                temp = ind_1
                ind_1 = ind_2
                ind_2 = temp

            pqr_twoblob = np.transpose(np.vstack((b1, b2)))
            xyz_twoblob = self._pqr_to_xyz(pqr_twoblob)
            xyz_twoblob_candidate.append(xyz_twoblob)
            twoblob_candidate_ind.append([ind_1, ind_2])

            i += 2 + ref
            neighbor_ind += 1
#       (return twoblob_candidate_ind, xyz_twoblob_candidate, unassigned_ind)
        #subfunction: match xyz_twoblob_candidate with predicted_blobs (input twoblob_candidate_ind, xyz_twoblob_candidate, unassigned_ind)
        #print("neighbor_ind",neighbor_ind)
        twoblob_new_ind_set = set(range(neighbor_ind))

        if xyz_twoblob_candidate:
            if predicted_blobs: #match predicted blobs and detected blobs
                xyz_led1_candidate = np.array([coord[:,0] for coord in xyz_twoblob_candidate])
                xyz_led2_candidate = np.array([coord[:,1] for coord in xyz_twoblob_candidate])

                xyz_led1_predicted = np.array([coord[:,0] for coord in predicted_blobs])
                xyz_led2_predicted = np.array([coord[:,1] for coord in predicted_blobs])

                led1_dist = cdist(xyz_led1_candidate, xyz_led1_predicted, 'euclidean')
                led2_dist = cdist(xyz_led2_candidate, xyz_led2_predicted, 'euclidean')
                dist_thresh = 300 #[mm] #500 before

                led1_dist = np.clip(led1_dist, 0, dist_thresh) #clip to threshold
                led2_dist = np.clip(led2_dist, 0, dist_thresh) #clip to threshold

                #add up normalized costs of both
                dist_normalized = (led1_dist + led2_dist)/(2*dist_thresh)
                #print("dist_normalized",dist_normalized)
                xyz_twoblob_matched_ind, track_twoblob_matched_ind = linear_sum_assignment(dist_normalized)
                xyz_twoblob_matched_ind = list(xyz_twoblob_matched_ind)
                track_twoblob_matched_ind = list(track_twoblob_matched_ind)

                #ignore matches with too high cost
                #print("cost",dist_normalized[xyz_twoblob_matched_ind, track_twoblob_matched_ind])
                for i, j in zip(xyz_twoblob_matched_ind, track_twoblob_matched_ind):
                    if dist_normalized[i,j] < 0.5: #.1-.5 of 500mm before
                        unassigned_ind.difference_update(twoblob_candidate_ind[i]) #the blobs from those indices are assigned now -> remove them from unassigned_ind
                    else:
                        xyz_twoblob_matched_ind.remove(i)
                        track_twoblob_matched_ind.remove(j)

                #(return xyz_twoblob_matched_ind, track_twoblob_matched_ind, unassigned_ind, xyz_twoblob_new_ind)

                #subfunction: search 3rd led for matched vertical pairs (input: xyz_twoblob_matched_ind, track_twoblob_matched_ind, unassigned_ind, predicted_blobs, xyz_twoblob_candidate)
                #now optimal assignment, not greedy
                cost_thresh = 0.3 #pw tune, but better smaller than 0.33 to not allow phi to cross phi_thresh
                phi_thresh = pi/4
                cost_weights = np.array([1, 1, 1], dtype=float) #hor, vert, phi; change if desired
                cost_weights /= np.sum(cost_weights)
                unassigned_ind_list = list(unassigned_ind)
                cost_mat = np.empty((len(xyz_twoblob_matched_ind), len(unassigned_ind_list)))
                for idx_i, i in enumerate(xyz_twoblob_matched_ind):
                    xyz_twoblob = xyz_twoblob_candidate[i]
                    phi_pred = predicted_phi[track_twoblob_matched_ind[idx_i]]
                    #find 3rd fitting led
                    for idx_j, j in enumerate(unassigned_ind_list): #take all unassigned_ind indices
                        pqr_b3 = all_blobs[:,j]
                        xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                        xyz_b3 = xyz_threeblob[:,2]
                        phi = self._orientation(xyz_threeblob)
                        led_hor_dist = sqrt((xyz_b3[0]-xyz_twoblob[0,0])**2 + (xyz_b3[1]-xyz_twoblob[1,0])**2)
                        led_vert_dist = abs(xyz_b3[2]-xyz_twoblob[2,0])
                        phi_dist = abs(np.arctan2(sin(phi-phi_pred), cos(phi-phi_pred)))
                        #normalize
                        led_hor_dist = abs(led_hor_dist-U_LED_DX)/U_LED_DX
                        led_vert_dist = led_vert_dist/U_LED_DZ
                        phi_dist = phi_dist/phi_thresh
                        cost_mat[idx_i, idx_j] = np.dot(cost_weights, np.array([np.clip(led_hor_dist, 0, 1), np.clip(led_vert_dist, 0, 1), np.clip(phi_dist, 0, 1)]))
                        #print(j, "predb3 to xyz_b3 norm", np.linalg.norm(xyz_b3_predicted - xyz_b3), "vert", led_vert_dist/U_LED_DZ, "hor", (np.abs(led_hor_dist-U_LED_DX)/U_LED_DX))
                duplet_matched_idx, thirdblob_matched_idx = linear_sum_assignment(cost_mat)
                for idx_i, idx_j in zip(duplet_matched_idx, thirdblob_matched_idx):
                    if cost_mat[idx_i, idx_j] < cost_thresh:
                        i = xyz_twoblob_matched_ind[idx_i]
                        j = unassigned_ind_list[idx_j]
                        xyz_twoblob = xyz_twoblob_candidate[i]
                        pqr_b3 = all_blobs[:,j]
                        xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                        phi = self._orientation(xyz_threeblob)
                        xyz_threeblob_matched.append(xyz_threeblob)
                        blob3_matched_ind.append([twoblob_candidate_ind[i][0],twoblob_candidate_ind[i][1],j])
                        xyz_threeblob_matched_ind.append(i)
                        track_threeblob_matched_ind.append(track_twoblob_matched_ind[idx_i])
                        phi_matched.append(phi)
                        unassigned_ind.discard(j)
                        j_neighbor_ind = np.argwhere([j in x for x in twoblob_candidate_ind])
                        if any(j_neighbor_ind): #blob j has been assigned now, but it also belongs to another twoblob pair
                            unassigned_ind.difference_update(twoblob_candidate_ind[j_neighbor_ind[0,0]]) # remove the blob that has been paired with j, cause it's probably its reflection
                            twoblob_new_ind_set.discard(j_neighbor_ind[0,0])

            #subfunction: second round to find 3rd led for xyz_twoblob_new: the ones that havent matched with an existing track and want to start a new track (input)
            twoblob_new_ind_set.difference_update(xyz_threeblob_matched_ind)
            xyz_twoblob_new_ind = list(twoblob_new_ind_set)
            cost_thresh = 0.2#  0.00001#pw tune, in noisefree world:
            cost_weights = np.array([1, 1], dtype=float) #hor, vert; change if desired
            cost_weights /= np.sum(cost_weights)
            cost_mat = np.empty((len(xyz_twoblob_new_ind), len(unassigned_ind)))
            unassigned_ind_list = list(unassigned_ind)

            #fill cost matrix to find 3rd led
            for idx_i, i in enumerate(xyz_twoblob_new_ind):
                xyz_twoblob = xyz_twoblob_candidate[i]
                for idx_j, j in enumerate(unassigned_ind_list):
                    if j in xyz_twoblob: #blob j already needed for my duplet -> impossible assignment, 'inf' cost
                        cost_mat[idx_i, idx_j] = 10000
                        continue
                    pqr_b3 = all_blobs[:,j]
                    xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                    xyz_b3 = xyz_threeblob[:,2]
                    led_hor_dist = sqrt((xyz_b3[0]-xyz_twoblob[0,0])**2 + (xyz_b3[1]-xyz_twoblob[1,0])**2)
                    led_vert_dist = abs(xyz_b3[2]-xyz_twoblob[2,0])
                    #normalize
                    led_hor_dist = abs(led_hor_dist-U_LED_DX)/U_LED_DX
                    led_vert_dist = led_vert_dist/U_LED_DZ
                    cost_mat[idx_i, idx_j] = np.dot(cost_weights, np.array([np.clip(led_hor_dist, 0, 1), np.clip(led_vert_dist, 0, 1)]))
            #assign cost
            duplet_matched_idx, thirdblob_matched_idx = linear_sum_assignment(cost_mat)
            for idx_i, idx_j in zip(duplet_matched_idx, thirdblob_matched_idx):
                if cost_mat[idx_i, idx_j] < cost_thresh:
                    i = xyz_twoblob_new_ind[idx_i]
                    j = unassigned_ind_list[idx_j]
                    xyz_twoblob = xyz_twoblob_candidate[i]
                    pqr_b3 = all_blobs[:,j]
                    xyz_threeblob = self._pqr_3_to_xyz(xyz_twoblob, pqr_b3)
                    phi = self._orientation(xyz_threeblob)
                    xyz_threeblob_new.append(xyz_threeblob)
                    xyz_threeblob_new_ind.append(i)
                    phi_new.append(self._orientation(xyz_threeblob))


        #this function evaluates the percentage of correctly matched leds

        self.environment.count_wrong_parsing(np.array(twoblob_candidate_ind)[xyz_twoblob_matched_ind], blob3_matched_ind)
        #self.environment.count_wrong_parsing(np.array(twoblob_candidate_ind), blob3_matched_ind)
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

    def move(self, detected_blobs, duration):
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

        phi_des = center_orient
        v_des = 0#0.2
        self.home_orient(phi_des, v_des)
        #self.depth_ctrl_vert(center_pos[2])
        z_des = 500 #+ self.id *100 #so that they are in different heights
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
