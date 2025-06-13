import numpy as np

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize the PID controller.

        Parameters:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_value: float, target_value: float, dt: float) -> float:
        """
        Compute the PID controller output.

        Parameters:
            current_value (float): The current measurement.
            target_value (float): The desired setpoint.
            dt (float): Time step.

        Returns:
            float: Control output.
        """
        error = target_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class QuadCopterController:
    def __init__(self, state: dict,
                 kp_pos: float, ki_pos: float, kd_pos: float,
                 kp_alt: float, ki_alt: float, kd_alt: float,
                 kp_att: float, ki_att: float, kd_att: float,
                 kp_yaw: float, ki_yaw: float, kd_yaw: float,
                 m: float, g: float, b: float,
                 u1_limit: float = 100.0, u2_limit: float = 10.0, 
                 u3_limit: float = 10.0, u4_limit: float = 10.0,
                 max_angle_deg: float = 30):
        """
        Initialize the quadcopter controller with PID controllers for position, altitude, attitude, and yaw.

        Parameters:
            state (dict): Current state of the drone.
            kp_pos, ki_pos, kd_pos (float): PID gains for position.
            kp_alt, ki_alt, kd_alt (float): PID gains for altitude.
            kp_att, ki_att, kd_att (float): PID gains for attitude (roll & pitch).
            kp_yaw, ki_yaw, kd_yaw (float): PID gains for yaw.
            m (float): Mass of the drone.
            g (float): Gravitational acceleration.
            b (float): Thrust coefficient.
            u1_limit, u2_limit, u3_limit, u4_limit (float): Saturation limits for the control commands.
            max_angle_deg (float): Maximum tilt angle in degrees.
        """
        self.u1_limit = u1_limit
        self.u2_limit = u2_limit
        self.u3_limit = u3_limit
        self.u4_limit = u4_limit

        # PID for position (x, y, z)
        self.pid_x = PIDController(kp_pos, ki_pos, kd_pos)
        self.pid_y = PIDController(kp_pos, ki_pos, kd_pos)
        self.pid_z = PIDController(kp_alt, ki_alt, kd_alt)
        
        # PID for attitude (roll and pitch) and a separate PID for yaw
        self.pid_roll  = PIDController(kp_att, ki_att, kd_att)
        self.pid_pitch = PIDController(kp_att, ki_att, kd_att)
        self.pid_yaw   = PIDController(kp_yaw, ki_yaw, kd_yaw)
        
        self.state = state
        self.m = m
        self.g = g
        self.b = b
        self.max_angle = np.radians(max_angle_deg)

    def update(self, state: dict, target: dict, dt: float) -> tuple:
        """
        Compute the control commands for the quadcopter.

        Parameters:
            state (dict): Current state of the drone.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.

        Returns:
            tuple: (thrust_command, roll_command, pitch_command, yaw_command)
        """
        x, y, z = state['pos']
        roll, pitch, yaw = state['angles']
        x_t, y_t, z_t = target['x'], target['y'], target['z']

        # Outer loop: position control with feed-forward for hover
        compensation = np.clip(1.0 / (np.cos(pitch) * np.cos(roll)), 1.0, 1.5) # Compensation factor for hover thrust
        hover_thrust = self.m * self.g * compensation # Hover thrust based on mass and gravity
        pid_z_output = self.pid_z.update(z, z_t, dt) # Altitude control output
        thrust_command = hover_thrust + pid_z_output # Total thrust command including altitude control

        pitch_des = np.clip(self.pid_x.update(x, x_t, dt), -self.max_angle, self.max_angle) # Desired pitch based on x position
        roll_des  = np.clip(-self.pid_y.update(y, y_t, dt), -self.max_angle, self.max_angle) # Desired roll based on y position
        
        # Compute desired yaw based on target position
        dx = target['x'] - x
        dy = target['y'] - y
        yaw_des = np.arctan2(dy, dx)
        
        # Inner loop: attitude control
        roll_command = self.pid_roll.update(roll, roll_des, dt) # Roll command based on desired roll
        pitch_command = self.pid_pitch.update(pitch, pitch_des, dt) # Pitch command based on desired pitch
        yaw_command = self.pid_yaw.update(yaw, 0, dt)  # Yaw command based on desired yaw ## (Alternatively, use yaw_des to rotate towards the target)

        # Saturate the commands
        thrust_command = np.clip(thrust_command, 0, self.u1_limit) 
        roll_command = np.clip(roll_command, -self.u2_limit, self.u2_limit)
        pitch_command = np.clip(pitch_command, -self.u3_limit, self.u3_limit)
        yaw_command = np.clip(yaw_command, -self.u4_limit, self.u4_limit)
        
        return (thrust_command, roll_command, pitch_command, yaw_command)