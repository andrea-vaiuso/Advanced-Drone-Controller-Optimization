# Author: Andrea Vaiuso
# Version: 2.0
# Date: 16.07.2025
# Description: This module defines the QuadcopterModel class, which simulates the dynamics of a quadcopter drone.
# It includes methods for translational and rotational dynamics, state updates, and wind effects.

import numpy as np
from Controller import QuadCopterController
from utils import wrap_angle

class QuadcopterModel:
    def __init__(self, m: float, I: np.ndarray, b: float, d: float, l: float, Cd: float, 
                Ca: np.ndarray, Jr: float,
                init_state: dict, controller: QuadCopterController,
                max_rpm: float = 5000.0, R: float = 0.1):
        """
        Initialize the physical model of the quadcopter.

        Parameters:
            m (float): Mass.
            I (np.ndarray): Moment of inertia vector.
            b (float): Thrust coefficient.
            d (float): Drag coefficient.
            l (float): Distance from the center to the rotor.
            Cd (float): Drag coefficient for the rotor.
            Ca (np.ndarray): Aerodynamic drag coefficients.
            Jr (float): Rotor inertia.
            init_state (dict): Initial state of the quadcopter.
            controller (QuadCopterController): Controller for the quadcopter.
            max_rpm (float): Maximum RPM for the motors. Default is 5000 RPM.
            R (float): Rotor radius in meters. Default is 0.1 m.
        """

        self.rho = 1.225  # Air density in kg/m³
        self.A = R**2 * np.pi  # Rotor disk area in m²
        self.g = 9.81

        self.m = m
        self.I = I
        self.b = b
        self.d = d
        self.l = l
        self.Cd = Cd
        self.Ca = Ca
        self.Jr = Jr
        self.R = R
        self.state = init_state
        self.init_state = init_state.copy()  # Store the initial state for reset
        self.controller = controller
        self.max_rpm = max_rpm
        self.delta_b = 0.0
        self.thrust = 0.0 
        self.thrust_no_wind = 0.0  # Thrust without wind effect
        
        self.max_rpm_sq = (self.max_rpm * 2 * np.pi / 60)**2 # Maximum RPM squared for clipping
        self.hover_rpm = self._compute_hover_rpm()

    def _compute_hover_rpm(self) -> None:
        """
        Compute the RPM value needed for hovering flight.
        """
        T_hover = self.m * self.g
        w_hover = np.sqrt(T_hover / (4 * self.b))
        rpm_hover = w_hover * 60.0 / (2.0 * np.pi)
        return rpm_hover
        # Uncomment the following line for debug information:
        # print(f"[INFO] Hover thrust needed = {T_hover:.2f} N, hover rpm per motor ~ {rpm_hover:.1f} rpm")

    def __str__(self) -> str:
        """
        Return a string representation of the quadcopter model.
        """
        return f"Quadcopter Model: state = {self.state}"

    def _translational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the translational accelerations.

        Parameters:
            state (dict): Current state.

        Returns:
            np.ndarray: Acceleration vector [x_ddot, y_ddot, z_ddot].
        """
        omega = self._rpm_to_omega(state['rpm'])
        x_dot, y_dot, z_dot = state['vel']
        roll, pitch, yaw = state['angles']

        # Total thrust from all motors plus any additional thrust from wind
        # In a more complex model, this should be calculated based on the blade element theory or similar
        self.thrust = (self.b + self.delta_b) * np.sum(np.square(omega))
        self.thrust_no_wind = self.b * np.sum(np.square(omega))  # Thrust without wind effect
        
        v = np.linalg.norm(state['vel'])

        # Compute drag force if the velocity is non-zero
        if v > 0:
            drag_magnitude = 0.5 * self.rho * self.A * self.Cd * v**2
            drag_vector = drag_magnitude * (state['vel'] / v)
        else:
            drag_vector = np.array([0.0, 0.0, 0.0])

        x_ddot = (self.thrust / self.m *
                  (np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll))
                  - drag_vector[0] / self.m)
        y_ddot = (self.thrust / self.m *
                  (np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll))
                  - drag_vector[1] / self.m)
        z_ddot = (self.thrust / self.m *
                  (np.cos(pitch) * np.cos(roll))
                  - drag_vector[2] / self.m - self.g)

        return np.array([x_ddot, y_ddot, z_ddot])

    def _rotational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the rotational accelerations.

        Parameters:
            state (dict): Current state.

        Returns:
            np.ndarray: Angular acceleration vector [phi_ddot, theta_ddot, psi_ddot].
        """
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']

        # Adding the wind effect to the thrust coefficient
        tot_b = (self.b + self.delta_b)
        
        roll_torque = self.l * tot_b * (omega[3]**2 - omega[1]**2)
        pitch_torque = self.l * tot_b * (omega[2]**2 - omega[0]**2)
        yaw_torque = self.d * (omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])

        phi_ddot = (roll_torque / self.I[0]
                    - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                    - Omega_r / self.I[0] * theta_dot
                    - (self.I[2] - self.I[1]) / self.I[0] * theta_dot * psi_dot)
        theta_ddot = (pitch_torque / self.I[1]
                      - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                      + Omega_r / self.I[1] * phi_dot
                      - (self.I[0] - self.I[2]) / self.I[1] * phi_dot * psi_dot)
        psi_ddot = (yaw_torque / self.I[2]
                    - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                    - (self.I[1] - self.I[0]) / self.I[2] * phi_dot * theta_dot)

        return np.array([phi_ddot, theta_ddot, psi_ddot])
    
    @staticmethod
    def _rpm_to_omega(rpm: np.ndarray) -> np.ndarray:
        """
        Convert motor RPM to angular velocity (rad/s).

        Parameters:
            rpm (np.ndarray): Array of motor RPMs.

        Returns:
            np.ndarray: Angular velocities in rad/s.
        """
        return rpm * 2 * np.pi / 60
    
    @staticmethod
    def _omega_to_rpm(omega: np.ndarray) -> np.ndarray:
        """
        Convert angular velocity (rad/s) to motor RPM.

        Parameters:
            omega (np.ndarray): Array of angular velocities in rad/s.

        Returns:
            np.ndarray: Motor RPMs.
        """
        return omega * 60 / (2 * np.pi)
    
    def _mixer(self, u1: float, u2: float, u3: float, u4: float) -> tuple:
        """
        Compute the RPM for each motor based on the control inputs.

        Parameters:
            u1, u2, u3, u4 (float): Control inputs.

        Returns:
            tuple: RPM values for each motor.
        """
        b = self.b
        d = self.d
        l = self.l
        
        w1_sq = (u1 / (4 * b)) - (u3 / (2 * b * l)) + (u4 / (4 * d))
        w2_sq = (u1 / (4 * b)) - (u2 / (2 * b * l)) - (u4 / (4 * d))
        w3_sq = (u1 / (4 * b)) + (u3 / (2 * b * l)) + (u4 / (4 * d))
        w4_sq = (u1 / (4 * b)) + (u2 / (2 * b * l)) - (u4 / (4 * d))
        
        w1_sq = np.clip(w1_sq, 0.0, self.max_rpm_sq)
        w2_sq = np.clip(w2_sq, 0.0, self.max_rpm_sq)
        w3_sq = np.clip(w3_sq, 0.0, self.max_rpm_sq)
        w4_sq = np.clip(w4_sq, 0.0, self.max_rpm_sq)
        
        w1 = np.sqrt(w1_sq)
        w2 = np.sqrt(w2_sq)
        w3 = np.sqrt(w3_sq)
        w4 = np.sqrt(w4_sq)
        
        rpm1 = w1 * 60.0 / (2.0 * np.pi)
        rpm2 = w2 * 60.0 / (2.0 * np.pi)
        rpm3 = w3 * 60.0 / (2.0 * np.pi)
        rpm4 = w4 * 60.0 / (2.0 * np.pi)

        return rpm1, rpm2, rpm3, rpm4

    def _rk4_step(self, state: dict, dt: float) -> dict: #Check physical meaning
        """
        Performs a single integration step using the classical 4th-order Runge-Kutta (RK4) method.
        The RK4 method is a numerical technique for solving ordinary differential equations (ODEs).
        It estimates the state of the system at the next time step by computing four increments (k1, k2, k3, k4),
        each representing the derivative of the state at different points within the interval. These increments
        are combined to produce a weighted average, providing a more accurate estimate than simpler methods
        like Euler integration.
        The state is represented as a dictionary containing position ('pos'), velocity ('vel'), angles ('angles'),
        angular velocity ('ang_vel'), and motor RPM ('rpm'). The function advances the state by time step `dt`
        using the system's translational and rotational dynamics.

        Parameters:
            state (dict): Current state.
            dt (float): Time step.

        Returns:
            dict: New state after the integration step.
        """
        def f(s: dict) -> dict:
            return {
                'pos': s['vel'],
                'vel': self._translational_dynamics(s),
                'angles': s['ang_vel'],
                'ang_vel': self._rotational_dynamics(s)
            }
        
        k1 = f(state)
        state1 = {key: state[key] + k1[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state1['rpm'] = state['rpm']
        k2 = f(state1)
        
        state2 = {key: state[key] + k2[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state2['rpm'] = state['rpm']
        k3 = f(state2)
        
        state3 = {key: state[key] + k3[key] * dt for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state3['rpm'] = state['rpm']
        k4 = f(state3)
        
        new_state = {}
        for key in ['pos', 'vel', 'angles', 'ang_vel']:
            new_state[key] = state[key] + (dt / 6) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        new_state['rpm'] = state['rpm']
        
        return new_state
    
    def update_wind(self, V: float, simulate_wind=True) -> None:
        """
        Update the wind signal for the quadcopter model.

        Parameters:
            V (float): Wind speed in m/s.
            simulate_wind (bool): Whether to simulate wind effects. Default is True.
        """
        if simulate_wind: 
            theta_0 = 2 * np.pi / 180  # Initial angle in radians
            omega = self._rpm_to_omega(self.hover_rpm)
            self.delta_b = (3/2) * (self.b / ((theta_0 * omega * self.R) + 1e-6)) * V #Need to be checked
        else: self.delta_b = 0.0


    def update_state(self, state: dict, target: dict, dt: float, ground_control: bool = True, hit_accel_threshold: float = 1.0) -> None:
        """
        Update the drone's state by computing control commands, mixing motor RPMs,
        and integrating the dynamics.

        Parameters:
            state (dict): Current state.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.
            ground_control (bool): Whether to apply ground control logic. Default is True.
            hit_accel_threshold (float): Threshold for detecting a hard landing. Default is 1.0 m/s² following MIL-STD-1290A.
        """
         # Control inputs from the controller
        u1, u2, u3, u4 = self.controller.update(state, target, dt, self.m)

        # Compute RPMs for each motor using the mixer function
        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4) 
        
        # Update the state with the new RPMs
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4]) 

        # Perform a Runge-Kutta 4th order integration step to update the state
        state = self._rk4_step(state, dt) 

        # Ensure the state is within valid ranges
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']]) 

        # Update the thrust in the state
        state['thrust'] = self.thrust  

        # Ground control logic
        if state['pos'][2] <= 0 and ground_control:
            state['pos'][2] = 0
            state['vel'][2] = 0  # Reset vertical velocity to zero
            # check if vertical acceleration is too high and differenciate from landing to hit
            if state['vel'][2] < -hit_accel_threshold:  # If the vertical velocity is exceeds the threshold
                print("[WARNING] Drone has hit the ground")
            else:
                print("[INFO] Drone has landed.")
        
        self.state = state  # Update the internal state

    def reset_state(self) -> None:
        """
        Reset the drone's state to the initial state.
        """
        self.state = self.init_state.copy()
        self.delta_b = 0.0
