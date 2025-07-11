import numpy as np
from Drone import QuadcopterModel
from World import World
from functools import reduce
from Wind import dryden_response
from matplotlib import pyplot as plt
from Noise.DNNModel import RotorSoundModel
import time

MIN_HEIGHT_FROM_GROUND = 1e-4  # Minimum height from ground to avoid singularities in noise calculations

# --- Main: Simulation and Plotting ---

class Simulation:
    def __init__(self, drone: QuadcopterModel, world:World, waypoints: list, 
                 dt=0.007, max_simulation_time=200.0, frame_skip=8, 
                 target_reached_threshold=2.0, dynamic_target_shift_threshold_prc=0.7,
                 noise_model: RotorSoundModel = None, noise_annoyance_radius=100):
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
            noise_model (np.ndarray): Optional noise model for simulating drone noise emissions.
            noise_annoyance_radius (int): Radius around the drone to consider for noise emissions.

        This simulation implements a dynamic target strategy where the drone follows a moving target
        along a path defined by waypoints. The target is computed dynamically based on the drone's position
        and the desired speed for each segment. The drone's state is updated using a Runge-Kutta integration method.
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
        self.world = world
        self.wind_signals = []
        self.simulate_wind = False
        self.noise_model = noise_model
        self.noise_annoyance_radius = noise_annoyance_radius
        self.simulation_time = 0.0

    def setWind(self, max_simulation_time, dt, height=100, airspeed=10, turbulence_level=30, axis=['u','v','w'], plot_wind_signal=False):
        """
        Set the wind conditions for the simulation using a Dryden wind model.
        Parameters:
            max_simulation_time (float): Maximum simulation time in seconds.
            dt (float): Time step for the simulation.
            height (float): Height above ground level for the wind model.
            airspeed (float): Airspeed of the drone.
            turbulence_level (int): Level of turbulence to simulate.
            plot_wind_signal (bool): If True, plot the generated wind signal for debugging.
        
        This method generates a wind signal using the Dryden wind model, which simulates atmospheric turbulence.
        The wind signal is generated for the w-axis (vertical) and can be used to simulate wind effects on the drone.
        """
        num_steps = int(max_simulation_time / dt)
        if type(axis) is str:
            axis = [axis]
        for ax in axis:
            self.wind_signals.append(
                dryden_response(axis=ax, height=height, airspeed=airspeed, turbulence_level=turbulence_level, time_steps=num_steps)
            )  # Initialize wind signals for each axis

        # Plot wind signals for debugging using subplots
        if plot_wind_signal:
            fig, axs = plt.subplots(len(axis), 1, figsize=(10, 3 * len(axis)))
            for i, ax in enumerate(axis):
                axs[i].plot(np.arange(num_steps) * dt, self.wind_signals[i])
                axs[i].set_title(f"Wind Signal for Axis {ax}")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel("Wind Speed")
            plt.tight_layout()
            plt.show()

        self.simulate_wind = True

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

        This method computes a dynamic target point along a segment defined by seg_start and seg_end.
        The target is determined by projecting the current drone position onto the segment and adding a look-ahead distance based on the desired speed. 
        The progress is calculated as the ratio of the target length to the segment length.
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


    def startSimulation(self, stop_at_target=True) -> tuple:
        """
        Start the simulation of the drone following dynamic targets along the waypoints.
        Parameters:
            stop_at_target (bool): If True, stop the simulation when the final target is reached.
        Returns:
            tuple: (positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, spl_history, swl_history, targets)

        This method initializes the dynamic target strategy, computes the moving target for each segment,
        and updates the drone's state at each time step. The drone follows the waypoints dynamically,
        adjusting its target based on its current position and the desired speed for each segment.
        The simulation runs for a maximum duration defined by max_simulation_time, and data is collected
        at specified intervals defined by frame_skip.
        """
        # --- Initialize time
        self.simulation_time = 0.0
        # --- Start timer
        t_0 = time.time()

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
        spl_history = [] 
        swl_history = []

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

            # Update the drone state using the dynamic target
            self.drone.update_state(self.drone.state, {'x': target_dynamic[0], 'y': target_dynamic[1], 'z': target_dynamic[2]}, self.dt)
            # If wind simulation is enabled, apply wind effects
            self.drone.update_wind(self.wind_signals[2][step], simulate_wind=self.simulate_wind) # Only vertical wind is applied
                
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

                if self.noise_model is not None:
                    # Calculate noise emissions in the ground area of a certain radius around the drone
                    avg_swl = 0.0
                    avg_spl = 0.0
                    x_drone, y_drone, z_drone = self.drone.state['pos']
                    ground_areas, ground_parameters = self.world.get_areas_in_circle(x = int(x_drone), 
                                                                                     y = int(y_drone), 
                                                                                     height = MIN_HEIGHT_FROM_GROUND, 
                                                                                     radius = self.noise_annoyance_radius, 
                                                                                     include_areas_out_of_bounds = True)
                    for area in ground_areas:
                        x_area, y_area, _ = area 
                        distance = np.linalg.norm([x_drone,y_drone,z_drone] - np.array([x_area, y_area, MIN_HEIGHT_FROM_GROUND]))
                        zeta_angle = np.arctan2(abs(self.drone.state['pos'][2]), distance)
                        spl, swl = self.noise_model.get_noise_emissions(zeta_angle = zeta_angle, 
                                                                        rpms = self.drone.state['rpm'], 
                                                                        distance = distance)
                        avg_spl += spl
                        avg_swl += swl

                    swl_history.append(avg_swl / len(ground_areas) if ground_areas else 0.0)
                    spl_history.append(avg_spl / len(ground_areas) if ground_areas else 0.0)

            # Check if drone reached the final target (within threshold) and stop the simulation early
            final_target = np.array([self.waypoints[-1]['x'], self.waypoints[-1]['y'], self.waypoints[-1]['z']])
            if np.linalg.norm(self.drone.state['pos'] - final_target) < self.target_reached_threshold and stop_at_target:
                print(f"Final target reached at time: {current_time:.2f} s")
                break
            
        # --- End timer
        self.simulation_time = time.time() - t_0
        print(f"Simulation completed in {self.simulation_time:.2f} seconds.")

        return (np.array(positions), 
                np.array(angles_history), 
                np.array(rpms_history), 
                np.array(time_history), 
                np.array(horiz_speed_history), 
                np.array(vertical_speed_history),
                np.array(spl_history), 
                np.array(swl_history), 
                np.array(targets))