import numpy as np
from Controller import QuadCopterController
from utils import wrap_angle


class RotorModel:
    def __init__(
        self,
        blade_chord: float = 0.02,
        blade_count: int = 2,
        blade_sections: int = 20,
        pitch_angle_rad: float = 0.2,  # radians
        lift_slope: float = 2 * np.pi,  # rad^{-1}, default thin-airfoil theory
        C_d: float = 0.01,  # profile drag coefficient
        R: float = 0.1,  # rotor radius [m]
    ):
        """
        Initialize the rotor model parameters.

        Parameters:
            blade_chord (float): Chord length [m].
            blade_count (int): Number of blades.
            blade_sections (int): Discretization sections per blade.
            pitch_angle_rad (float): Blade pitch angle [rad].
            lift_slope (float): Lift-curve slope [per rad].
            C_d (float): Drag coefficient (profile).
            R (float): Rotor radius [m].
        """
        self.blade_chord = blade_chord
        self.blade_count = blade_count
        self.blade_sections = blade_sections
        self.pitch_angle_rad = pitch_angle_rad
        self.lift_slope = lift_slope
        self.C_d = C_d
        self.R = R
        self.disk_area = np.pi * R**2


class QuadcopterModel:
    def __init__(
        self,
        m: float,
        I: np.ndarray,
        d: float,
        l: float,
        Ca: np.ndarray,
        Jr: float,
        rotor_model: RotorModel,
        init_state: dict,
        controller: QuadCopterController,
        max_rpm: float = 5000.0,
    ):
        """
        Initialize the quadcopter physical model.

        Parameters:
            m (float): Mass [kg].
            I (np.ndarray): Moments of inertia [I_x, I_y, I_z].
            d (float): Yaw drag coefficient.
            l (float): Arm length [m].
            Ca (np.ndarray): Rotational damping coefficients [x, y, z].
            Jr (float): Rotor inertia [kg·m²].
            rotor_model (RotorModel): Rotor geometry and aero data.
            init_state (dict): Initial state dictionary.
            controller (QuadCopterController): Attitude controller instance.
            max_rpm (float): Maximum motor RPM.
        """
        # Physical constants
        self.rho = 1.225  # air density [kg/m³]
        self.g = 9.81     # gravity [m/s²]

        # Vehicle properties
        self.m = m
        self.I = I
        self.d = d
        self.l = l
        self.Ca = Ca
        self.Jr = Jr
        self.rotor_model = rotor_model

        # State and control
        self.state = init_state.copy()
        self.init_state = init_state.copy()
        self.controller = controller

        # Limits
        self.max_omega_sq = (max_rpm * 2 * np.pi / 60)**2

        # Thrust storage
        self.thrust = 0.0
        self.state['thrust'] = 0.0
        self.thrust_coeff = 0.0

    def __str__(self) -> str:
        return f"QuadcopterModel(state={self.state})"

    @staticmethod
    def _rpm_to_omega(rpm: np.ndarray) -> np.ndarray:
        """
        Convert RPM array to angular speed [rad/s].

        Parameters:
            rpm (np.ndarray): Motor speeds [RPM].

        Returns:
            np.ndarray: Angular speeds [rad/s].
        """
        return rpm * 2 * np.pi / 60

    @staticmethod
    def _omega_to_rpm(omega: np.ndarray) -> np.ndarray:
        """
        Convert angular speed array to RPM.
        """
        return omega * 60 / (2 * np.pi)

    def _compute_bemt_thrust(self, omega: np.ndarray) -> tuple:
        """
        Compute total thrust and average thrust coefficient via BEMT.

        Parameters:
            omega (np.ndarray): Rotor speeds [rad/s].

        Returns:
            (T_total [N], C_T_avg)
        """
        T_total = 0.0
        CT_list = []
        R = self.rotor_model.R
        dr = R / self.rotor_model.blade_sections

        for w in omega:
            # Thrust per single rotor
            T_rotor = 0.0
            for i in range(self.rotor_model.blade_sections):
                r = dr * (i + 0.5)
                Vt = w * r
                # hover assumption: axial inflow = 0
                phi = 0.0
                alpha = self.rotor_model.pitch_angle_rad - phi
                # aerodynamic coefficients
                Cl = self.rotor_model.lift_slope * alpha
                Cd = self.rotor_model.C_d
                # differential lift and drag
                q = 0.5 * self.rho * Vt**2
                dL = q * self.rotor_model.blade_chord * Cl * dr
                dD = q * self.rotor_model.blade_chord * Cd * dr
                # project to thrust
                dT = dL * np.cos(phi) - dD * np.sin(phi)
                T_rotor += dT
            # multiply by blade count
            T_rotor *= self.rotor_model.blade_count
            T_total += T_rotor
            # local thrust coefficient
            CT = T_rotor / (self.rho * self.rotor_model.disk_area * (w * R)**2 + 1e-12)
            CT_list.append(CT)

        CT_avg = float(np.mean(CT_list)) if CT_list else 0.0
        return T_total, CT_avg

    def _translational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute translational accelerations [m/s²].
        """
        omega = self._rpm_to_omega(state['rpm'])
        # compute thrust and coefficient
        T, CT = self._compute_bemt_thrust(omega)
        self.thrust = T
        self.state['thrust'] = T
        self.thrust_coeff = CT

        # drag
        v = np.linalg.norm(state['vel'])
        if v > 0.0:
            A_body = self.rotor_model.disk_area  # placeholder
            drag_mag = 0.5 * self.rho * A_body * self.rotor_model.C_d * v**2
            drag = drag_mag * (state['vel'] / v)
        else:
            drag = np.zeros(3)

        roll, pitch, yaw = state['angles']
        # thrust vector in body frame, assume thrust along body z-axis
        thrust_body = np.array([0, 0, T])
        # rotation to inertial frame
        R_ib = self._rotation_matrix(roll, pitch, yaw)
        thrust_inertial = R_ib @ thrust_body

        acc = thrust_inertial / self.m - drag / self.m + np.array([0, 0, -self.g])
        return acc

    def _rotational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute rotational accelerations [rad/s²].
        """
        omega = self._rpm_to_omega(state['rpm'])
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        roll_torque = self.l * self.thrust_coeff * (
            omega[3]**2 - omega[1]**2)
        pitch_torque = self.l * self.thrust_coeff * (
            omega[2]**2 - omega[0]**2)
        yaw_torque = self.d * (
            omega[0]**2 - omega[1]**2 + omega[2]**2 - omega[3]**2)
        Omega_r = self.Jr * (omega[0] - omega[1] + omega[2] - omega[3])

        phi_dd = (roll_torque / self.I[0]
                  - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                  - Omega_r * theta_dot / self.I[0]
                  - (self.I[2] - self.I[1]) * theta_dot * psi_dot / self.I[0])
        theta_dd = (pitch_torque / self.I[1]
                    - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                    + Omega_r * phi_dot / self.I[1]
                    - (self.I[0] - self.I[2]) * phi_dot * psi_dot / self.I[1])
        psi_dd = (yaw_torque / self.I[2]
                  - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                  - (self.I[1] - self.I[0]) * phi_dot * theta_dot / self.I[2])
        return np.array([phi_dd, theta_dd, psi_dd])

    def _rotation_matrix(self, phi, theta, psi) -> np.ndarray:
        """
        Compute body-to-inertial rotation matrix.
        """
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        # ZYX
        Rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
        Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
        Rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
        return Rz @ Ry @ Rx

    def _mixer(self, u1: float, u2: float, u3: float, u4: float) -> np.ndarray:
        """
        Map control inputs to motor RPMs.
        """
        # simple allocation using thrust-coefficient inverse
        # u1... total thrust, u2,u3 roll/pitch torques, u4 yaw torque
        ct_h = self.thrust_coeff + 1e-12
        w_sq = np.zeros(4)
        w_sq[0] = (u1/4 - u3/(2*self.l) + u4/(4*self.d)) / ct_h
        w_sq[1] = (u1/4 - u2/(2*self.l) - u4/(4*self.d)) / ct_h
        w_sq[2] = (u1/4 + u3/(2*self.l) + u4/(4*self.d)) / ct_h
        w_sq[3] = (u1/4 + u2/(2*self.l) - u4/(4*self.d)) / ct_h
        w_sq = np.clip(w_sq, 0.0, self.max_omega_sq)
        omega = np.sqrt(w_sq)
        return self._omega_to_rpm(omega)

    def _rk4_step(self, state: dict, dt: float) -> dict:
        """
        RK4 integration of state.
        """
        def f(s):
            return {
                'pos': s['vel'],
                'vel': self._translational_dynamics(s),
                'angles': s['ang_vel'],
                'ang_vel': self._rotational_dynamics(s)
            }
        k1 = f(state)
        s2 = {k: state[k] + k1[k]*dt/2 for k in k1}
        s2['rpm'] = state['rpm']
        k2 = f(s2)
        s3 = {k: state[k] + k2[k]*dt/2 for k in k2}
        s3['rpm'] = state['rpm']
        k3 = f(s3)
        s4 = {k: state[k] + k3[k]*dt for k in k3}
        s4['rpm'] = state['rpm']
        k4 = f(s4)
        new = {}
        for key in ['pos','vel','angles','ang_vel']:
            new[key] = state[key] + dt*(k1[key]+2*k2[key]+2*k3[key]+k4[key])/6
        new['rpm'] = state['rpm']
        return new

    def update_wind(self, V: float, simulate_wind: bool = True) -> None:
        """
        Placeholder: wind effect can adjust blade inflow angle.
        """
        # no wind effect in BEMT hover
        pass

    def update_state(self, state: dict, target: dict, dt: float,
                     ground_control: bool = True, hit_accel_threshold: float = 1.0) -> None:
        """
        Main update: compute control, mixer, integrate, enforce ground.
        """
        # controller outputs thrust/torques
        u1, u2, u3, u4 = self.controller.update(state, target, dt)
        # mixer to RPMs
        rpm_vals = self._mixer(u1, u2, u3, u4)
        state['rpm'] = rpm_vals
        # integrate
        state = self._rk4_step(state, dt)
        # wrap angles
        state['angles'] = np.array([wrap_angle(a) for a in state['angles']])
        # ground
        if state['pos'][2] <= 0 and ground_control:
            state['pos'][2] = 0
            state['vel'][2] = 0
            if state['vel'][2] < -hit_accel_threshold:
                print("[WARNING] Hard ground impact")
            else:
                print("[INFO] Landed safely")
        self.state = state

    def reset_state(self) -> None:
        """
        Reset to initial conditions.
        """
        self.state = self.init_state.copy()
