import numpy as np
from Controller import QuadCopterController
from utils import wrap_angle

class QuadcopterModel:
    def __init__(self, m: float, I: np.ndarray, b: float, d: float, l: float, 
                 Cd: np.ndarray, Ca: np.ndarray, Jr: float,
                 init_state: dict, controller: QuadCopterController,
                 g: float = 9.81, max_rpm: float = 5000.0, R: float = 0.1):
        """
        Initialize the physical model of the quadcopter.

        Parameters:
            m (float): Mass.
            I (np.ndarray): Moment of inertia vector.
            b (float): Thrust coefficient.
            d (float): Drag coefficient.
            l (float): Arm length.
            Cd (np.ndarray): Translational drag coefficients.
            Ca (np.ndarray): Rotational damping coefficients.
            Jr (float): Rotor inertia.
            init_state (dict): Initial state.
            controller (QuadCopterController): Controller instance.
            g (float): Gravitational acceleration.
            max_rpm (float): Maximum RPM.
            R (float): Rotor radius in meters.
        """
        self.m = m
        self.I = I
        self.b = b
        self.d = d
        self.l = l
        self.Cd = Cd
        self.Ca = Ca
        self.Jr = Jr
        self.g = g
        self.R = R
        self.state = init_state
        self.controller = controller
        self.max_rpm = max_rpm
        self.delta_b = 0.0
        
        self.max_rpm_sq = (self.max_rpm * 2 * np.pi / 60)**2
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
        thrust = (self.b + self.delta_b) * np.sum(np.square(omega))
        
        v = np.linalg.norm(state['vel'])
        rho = 1.225  # Air density in kg/m³
        A = 0.1      # Reference area in m²
        C_d = 0.47   # Drag coefficient

        if v > 0:
            drag_magnitude = 0.5 * rho * A * C_d * v**2
            drag_vector = drag_magnitude * (state['vel'] / v)
        else:
            drag_vector = np.array([0.0, 0.0, 0.0])

        x_ddot = (thrust / self.m *
                  (np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll))
                  - drag_vector[0] / self.m)
        y_ddot = (thrust / self.m *
                  (np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll))
                  - drag_vector[1] / self.m)
        z_ddot = (thrust / self.m *
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
    
    def _rpm_to_omega(self, rpm: np.ndarray) -> np.ndarray:
        """
        Convert motor RPM to angular velocity (rad/s).

        Parameters:
            rpm (np.ndarray): Array of motor RPMs.

        Returns:
            np.ndarray: Angular velocities in rad/s.
        """
        return rpm * 2 * np.pi / 60
    
    def _omega_to_rpm(self, omega: np.ndarray) -> np.ndarray:
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

    def _rk4_step(self, state: dict, dt: float) -> dict:
        """
        Perform one Runge-Kutta 4th order integration step.

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
            self.delta_b = (3/2) * (self.b / ((theta_0 * omega * self.R) + 1e-4)) * V
        else: self.delta_b = 0.0


    def update_state(self, state: dict, target: dict, dt: float, ground_control: bool = True) -> None:
        """
        Update the drone's state by computing control commands, mixing motor RPMs,
        and integrating the dynamics.

        Parameters:
            state (dict): Current state.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.

        """

        u1, u2, u3, u4 = self.controller.update(state, target, dt) # Control inputs from the controller

        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4) # Compute RPMs for each motor
        
        state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4]) # Update the state with the new RPMs

        state = self._rk4_step(state, dt) # Perform a Runge-Kutta 4th order integration step to update the state

        state['angles'] = np.array([wrap_angle(a) for a in state['angles']]) # Ensure the state is within valid ranges

        # If the drone hits the ground, reset its altitude to 0
        if state['pos'][2] < 0 and ground_control:
            state['pos'][2] = 0
            print("[WARNING] Drone has hit the ground! Resetting altitude to 0.")
        
        self.state = state  # Update the internal state