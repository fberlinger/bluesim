"""Helper class to simulate the dynamics of Bluebot.
"""
from math import *
import numpy as np

class Dynamics():
    """Simulates the dynamics of BlueBot with Euler integration according to its equations of motion.
    """

    def __init__(self, environment):
        self.environment = environment
        self.steps = 10
        
        # Robot Specs
        self.rho = 998 # [kg/m^3], water density
        self.l_robot = 0.150 # [m], including fin
        self.w_robot = 0.050 # [m]
        self.h_robot = 0.080 # [m]
        self.A_x = pi/4 * self.h_robot * self.w_robot # [m**2]
        self.A_y = pi/4 * self.l_robot * self.h_robot + 2 * 0.00075 # [m**2], including fins
        self.A_z = pi/4 * self.l_robot * self.w_robot # [m**2]
        self.A_phi = self.A_y
        self.m_robot = 2*0.25 # [kg], including added mass
        self.I_robot = self.m_robot/5 * 1/4*(self.l_robot**2 + self.h_robot**2) # [kg*m**2]
        self.C_dx_fwd = 0.5 # c.f. cone
        self.C_dx_bwd = 1.0 # c.f. cone
        self.C_dy_static = 2.1 # c.f. flat plate
        self.C_dz = 0.7
        self.C_dphi_static = 1.0
        self.pect_dist = 0.055 # [m]
        self.pect_angle = pi / 6 # [rad]
        self.F_buoy = 0.010 # [N]
        self.vx_max = 0.160 # [m/s]

        # Initialize Control
        self.F_caud = 0 # [N]
        self.F_PR = 0 # [N]
        self.F_PL = 0 # [N]
        self.F_dors = 0 # [N]

    def update_ctrl(self, dorsal, caudal, pect_r, pect_l):
        """Update BlueBots fin control. Those thrust forces are then used in the equations of motion.

        Args:
            dorsal (float): Dorsal gain
            caudal (float): Caudal gain
            pect_r (float): Pectoral right gain
            pect_l (float): Pectoral left gain
        """
        F_caud_max = 0.020 # [N]
        F_PR_max = 0.006 # [N]
        F_PL_max = 0.006 # [N]
        F_dors_max = 0.020 # [N]

        self.F_caud = caudal * F_caud_max
        self.F_PR = pect_r * F_PR_max
        self.F_PL = pect_l * F_PL_max
        self.F_dors = dorsal * F_dors_max

    def simulate_move(self, source_id, duration):
        """Simulates move starting from current global coordinates based on current velocities and fin control. Returns next global coordinates.
        """
        deltat = duration / self.steps

        mm_to_m = 1/1000 # millimeter (environment) to meter (here)
        m_to_mm = 1000 # meter (here) to millimeter (environment)

        # Conversion to Meter of Current Global pos and vel in Environment
        g_P_r = mm_to_m * self.environment.pos[source_id,:3]
        phi = self.environment.pos[source_id,3]
        g_Pdot_r = mm_to_m * self.environment.vel[source_id,:3]
        vphi = self.environment.vel[source_id,3]

        # Global to Robot Transformation
        r_T_g = self.environment.rot_global_to_robot(phi)
        r_Pdot_r = r_T_g @ g_Pdot_r
        vx = r_Pdot_r[0]
        vy = r_Pdot_r[1]
        vz = r_Pdot_r[2]

        for step in range(self.steps):
            # Equations of Motion
            x_dot = vx
            y_dot = vy
            z_dot = vz
            phi_dot = vphi

            self.C_dphi = self.C_dphi_static + self.C_dphi_static * 9 * abs(x_dot) / self.vx_max
            self.C_dy = self.C_dy_static + self.C_dy_static * 4 * abs(x_dot) / self.vx_max
            if x_dot > 0:
                self.C_dx = self.C_dx_fwd
            else:
                self.C_dx = self.C_dx_bwd

            vx_dot = 1/self.m_robot * (self.F_caud - sin(self.pect_angle)*self.F_PL - sin(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dx*self.A_x*np.sign(x_dot)*x_dot**2)
            vy_dot = 1/self.m_robot * (cos(self.pect_angle)*self.F_PL - cos(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dy*self.A_y*np.sign(y_dot)*y_dot**2)
            vz_dot = 1/self.m_robot * (self.F_dors - self.F_buoy - 1/2*self.rho*self.C_dz*self.A_z*np.sign(z_dot)*z_dot**2)
            vphi_dot = 1/self.I_robot * (self.pect_dist*cos(self.pect_angle)*self.F_PL - self.pect_dist*cos(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dphi*self.A_phi*np.sign(phi_dot)*(self.l_robot/6*phi_dot)**2)

            # Euler Integration
            vx = x_dot + deltat*vx_dot
            vy = y_dot + deltat*vy_dot
            vz = z_dot + deltat*vz_dot

            phi = phi + deltat*phi_dot
            vphi = phi_dot + deltat*vphi_dot

            # Robot to Global Transformation
            g_T_r = self.environment.rot_robot_to_global(phi)
            g_Pdot_r = g_T_r @ np.array([vx, vy, vz])
            g_P_r = g_P_r + deltat*np.transpose(g_Pdot_r)

        # Conversion to millimeter and appendage of phi and vphi
        pos = np.concatenate((m_to_mm * g_P_r, np.array([phi])), axis=0)
        vel = np.concatenate((m_to_mm * g_Pdot_r, np.array([vphi])), axis=0)

        return (pos, vel)