import numpy as np
from Drone import QuadcopterModel
from World import World
from functools import reduce

# --- Main: Simulation and Plotting ---

class Simulation:
    def __init__(self, drone: QuadcopterModel, world:World, waypoints: list, 
                 dt=0.007, max_simulation_time=200.0, frame_skip=8, 
                 target_reached_threshold=2.0, dynamic_target_shift_threshold_prc=0.7,
                 simulate_wind = True, min_freq_gust=0.1, max_freq_gust=2.0, max_gust=5.0):
        """
        Initialize the simulation with the drone model, world, waypoints, and parameters.
        Parameters:
            drone (QuadcopterModel): The drone model to simulate.
            world (World): The world in which the simulation takes place.
            waypoints (list): List of waypoints with 'x', 'y', 'z', and 'v' keys.
            dt (float): Time step for the simulation.
            max_simulation_time (float): Maximum simulation time in seconds.
            frame_skip (int): Number of steps to skip for data collection.
            target_reached_threshold (float): Threshold distance to consider the target reached.
            dynamic_target_shift_threshold_prc (float): Percentage of segment completion to shift target.
            simulate_wind (bool): Whether to simulate wind gusts.
            min_freq_gust (float): Minimum frequency for wind gust noise generation.
            max_freq_gust (float): Maximum frequency for wind gust noise generation.
            max_gust (float): Maximum gust strength.

        This simulation implements a dynamic target strategy where the drone follows a moving target
        along a path defined by waypoints. The target is computed dynamically based on the drone's position
        and the desired speed for each segment. The drone's state is updated using a Runge-Kutta integration method.
        The simulation also includes a band-limited noise model to simulate wind gusts affecting the drone's flight.
        The drone will follow the waypoints, adjusting its target dynamically based on its current position
        and the desired speed for each segment. The simulation stops when the drone reaches the final target
        within a specified threshold distance or when the maximum simulation time is reached.
        """

        self.dt = dt
        self.max_simulation_time = max_simulation_time
        self.num_steps = int(max_simulation_time / dt)
        self.frame_skip = frame_skip
        self.target_reached_threshold = target_reached_threshold
        self.dynamic_target_shift_threshold_prc = dynamic_target_shift_threshold_prc
        self.waypoints = waypoints
        self.drone = drone
        self.drone.simulate_wind = simulate_wind
        self.world = world

        if simulate_wind:
            self.wind_noise = self._band_limited_noise(
                min_freq=min_freq_gust, max_freq=max_freq_gust,
                samples=self.num_steps,
                samplerate=1/self.dt
            )
            self.max_gust = max_gust

    def _band_limited_noise(self, min_freq, max_freq, samples, samplerate):
        """
        Generate band-limited noise using a sum of sine waves with random phases.
        This simulates wind gusts affecting the drone's flight.

        Parameters:
            min_freq (float): Minimum frequency of the noise.
            max_freq (float): Maximum frequency of the noise.
            samples (int): Number of samples to generate.
            samplerate (float): Sampling rate in Hz.
        Returns:
            np.array: Array of band-limited noise samples.
        """
        t = np.linspace(0, samples/samplerate, samples)
        freqs = np.arange(min_freq, max_freq+1, samples/samplerate)
        phases = np.random.rand(len(freqs))*2*np.pi
        signals = [np.sin(2*np.pi*freq*t + phase)
                for freq, phase in zip(freqs, phases)]
        signal = reduce(lambda a, b: a+b, signals)
        signal /= np.max(signal)
        return signal

    def _compute_moving_target(self, drone_pos, seg_start, seg_end, v_des, k=1.0):
        """
        Compute the dynamic target point along a segment with a look-ahead distance of L = k*v_des.

        Parameters:
            drone_pos (np.array): Current drone position [x, y, z].
            seg_start (np.array): Start point of the segment.
            seg_end (np.array): End point of the segment.
            v_des (float): Desired speed for this segment.
            k (float): Scaling factor for the look-ahead distance.

        Returns:
            tuple: (target, progress) where target is the dynamic target point [x, y, z],
                and progress is the fraction of the segment covered.
        """
        seg_vector = seg_end - seg_start
        seg_length = np.linalg.norm(seg_vector)
        if seg_length == 0:
            return seg_end, 1.0
        seg_dir = seg_vector / seg_length

        # Project current position onto the segment
        proj_length = np.dot(drone_pos - seg_start, seg_dir)
        # Look-ahead distance
        L = k * v_des
        target_length = proj_length + L

        # Do not exceed the final waypoint
        if target_length > seg_length:
            target_length = seg_length
        target = seg_start + target_length * seg_dir
        progress = target_length / seg_length
        return target, progress


    def startSimulation(self):
        # --- Initialize Dynamic Target Strategy ---
        current_seg_idx = 0
        seg_start = self.drone.state['pos'].copy()
        seg_end = np.array([self.waypoints[current_seg_idx]['x'], 
                            self.waypoints[current_seg_idx]['y'], 
                            self.waypoints[current_seg_idx]['z']])
        v_des = self.waypoints[current_seg_idx]['v']
        k_lookahead = 1.0  # Scaling parameter for look-ahead distance

        # Lists for storing data for animation and plotting
        positions = []
        angles_history = []
        rpms_history = []
        time_history = []
        horiz_speed_history = []
        vertical_speed_history = []
        targets = []  # List to store the dynamic target

        num_steps = int(self.max_simulation_time / self.dt)

        # Simulation loop
        for step in range(num_steps):
            # Compute dynamic target along the current segment
            target_dynamic, progress = self._compute_moving_target(self.drone.state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
            
            # If progress on current segment is nearly complete, move to the next segment if available
            if progress >= self.dynamic_target_shift_threshold_prc:
                current_seg_idx += 1
                if current_seg_idx < len(self.waypoints):
                    seg_start = seg_end
                    seg_end = np.array([self.waypoints[current_seg_idx]['x'], 
                                        self.waypoints[current_seg_idx]['y'], 
                                        self.waypoints[current_seg_idx]['z']])
                    v_des = self.waypoints[current_seg_idx]['v']
                    target_dynamic, progress = self._compute_moving_target(self.drone.state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
                else:
                    target_dynamic = seg_end  # Final waypoint: fixed target
            
            gust_norm = self.wind_noise[step]
            gust_vel = np.array([gust_norm * self.max_gust, 0.0, 0.0])
            self.drone.wind_velocity = gust_vel

            # Update the drone state using the dynamic target
            self.drone.state = self.drone.update_state(self.drone.state, {'x': target_dynamic[0], 'y': target_dynamic[1], 'z': target_dynamic[2]}, self.dt)
            current_time = step * self.dt

            # Save data every 'frame_skip' steps
            if step % self.frame_skip == 0:
                positions.append(self.drone.state['pos'].copy())
                angles_history.append(self.drone.state['angles'].copy())
                rpms_history.append(self.drone.state['rpm'].copy())
                time_history.append(current_time)
                horiz_speed_history.append(np.linalg.norm(self.drone.state['vel'][:2]))
                vertical_speed_history.append(self.drone.state['vel'][2])
                targets.append(target_dynamic.copy())

            # Check if drone reached the final target (within threshold) and stop the simulation early
            final_target = np.array([self.waypoints[-1]['x'], self.waypoints[-1]['y'], self.waypoints[-1]['z']])
            if np.linalg.norm(self.drone.state['pos'] - final_target) < self.target_reached_threshold:
                print(f"Final target reached at time: {current_time:.2f} s")
                break

        return np.array(positions), np.array(angles_history), np.array(rpms_history), np.array(time_history), np.array(horiz_speed_history), np.array(vertical_speed_history), np.array(targets)