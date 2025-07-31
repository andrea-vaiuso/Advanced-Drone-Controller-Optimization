# Author: Andrea Vaiuso
# Version: 2.1
# Date: 28.07.2025
# Description: This module defines the Simulation class for simulating a drone's flight path with dynamic targets,
# wind effects, and noise emissions. It includes methods for setting wind conditions, computing dynamic targets,
# and running the simulation with data collection.

import numpy as np
from Drone import QuadcopterModel
from World import World
from Wind import dryden_response
from matplotlib import pyplot as plt
from Noise.DNNModel import RotorSoundModel
import time

MIN_HEIGHT_FROM_GROUND = 1e-4  # Minimum height from ground to avoid singularities in noise calculations

# --- Main: Simulation and Plotting ---

class Simulation:
    def __init__(self, drone: QuadcopterModel, world: World, waypoints: list,
                 dt: float = 0.007, max_simulation_time: float = 200.0, frame_skip: int = 8,
                 target_reached_threshold: float = 2.0,
                 target_shift_threshold_distance: float = 5,
                 noise_model: RotorSoundModel = None, noise_annoyance_radius: int = 30,
                 generate_sound_emission_map: bool = False):
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
            target_shift_threshold_distance (float): Distance to consider for shifting the target.
            noise_model (RotorSoundModel): Optional noise model for simulating drone noise emissions.
            noise_annoyance_radius (int): Radius around the drone to consider for noise emissions.
            generate_sound_emission_map (bool): If True, generate a sound map of noise emissions.

        This simulation implements a dynamic target strategy where the drone follows a moving target
        along a path defined by waypoints. The target is computed dynamically based on the drone's position
        and the desired speed for each segment. The drone's state is updated using a Runge-Kutta integration method.
        The simulation stops when the drone reaches the final target within a specified threshold distance
        or when the maximum simulation time is reached. Wind conditions can be simulated using a Dryden wind model by using the `setWind` method.
        """
        self.drone = drone
        self.world = world
        self.waypoints = waypoints
        self.max_target_length = [0.0] * len(self.waypoints)
        self.dt = dt
        self.max_simulation_time = max_simulation_time
        self.frame_skip = frame_skip
        self.target_reached_threshold = target_reached_threshold
        self.target_shift_threshold_distance = target_shift_threshold_distance
        self.noise_model = noise_model
        self.noise_annoyance_radius = noise_annoyance_radius
        

        # Wind simulation parameters
        self.wind_signals = []
        self.simulate_wind = False

        # Histories for simulation data
        self.positions = []
        self.angles_history = []
        self.rpms_history = []
        self.time_history = []
        self.horiz_speed_history = []
        self.vertical_speed_history = []
        self.targets = []
        self.spl_history = []
        self.swl_history = []
        self.thrust_history = []
        self.power_history = []
        self.delta_b_history = []
        self.thrust_no_wind_history = []
        self.distance_history = []
        self.seg_idx_history = []

        # Simulation runtime
        self.simulation_time = 0.0
        self.current_seg_idx = 0
        self.navigation_time = None
        self.has_moved = True
        self.has_reached_target = False

        self.generate_sound_emission_map = generate_sound_emission_map
        self.noise_emission_map = {}


    def startSimulation(self, stop_at_target: bool = True,
                        stop_sim_if_not_moving: bool = False,
                        use_static_target: bool = False,
                        verbose: bool = True):
        """
        Start the simulation of the drone following dynamic or static targets along the waypoints.
        Parameters:
            stop_at_target (bool): If True, stop when the final target is reached.
            verbose (bool): If True, print simulation progress and completion messages.
            stop_sim_if_not_moving (bool): If True, stop simulation if the drone is not moving for a certain period.
            use_static_target (bool): If True, use ``compute_static_target`` instead of dynamic targeting.

        This method updates the drone's state at each time step and stores data in class attributes.
        """
        # Reset runtime and histories
        self.simulation_time = 0.0
        
        # Clear previous histories
        self._clear_histories()

        # Reset drone state to initial conditions
        self.drone.reset_state() 

        # Initialize histories for dynamic targeting
        self.max_target_length = [0.0] * len(self.waypoints)

        # Start timer
        t_0 = time.time()

        # Initialize dynamic targeting
        
        seg_start = self.drone.state['pos'].copy()
        seg_end = np.array([self.waypoints[0]['x'], self.waypoints[0]['y'], self.waypoints[0]['z']])
        v_des = self.waypoints[0]['v']
        k_lookahead = 1.0
        num_steps = int(self.max_simulation_time / self.dt)

        # Main simulation loop
        for step in range(num_steps):
            if use_static_target:
                target_dynamic, self.current_seg_idx, seg_end = self._compute_static_target(
                    self.current_seg_idx, seg_end)
            else:
                target_dynamic, self.current_seg_idx, seg_start, seg_end, v_des = self._compute_dynamic_target(
                    self.current_seg_idx, seg_start, seg_end, v_des, k_lookahead)
            
            # Update wind if set
            if self.simulate_wind and len(self.wind_signals) >= 3:
                wind_xyz_signal = np.array([self.wind_signals[0][step],
                                            self.wind_signals[1][step],
                                            self.wind_signals[2][step]])
                self.drone.update_wind(wind_xyz_signal, simulate_wind=True)

            # Update drone state
            self.drone.update_state({'x': target_dynamic[0], 'y': target_dynamic[1], 'z': target_dynamic[2]},
                                     self.dt, verbose=False)

            
            current_time = step * self.dt

            # Store data at specified intervals
            if step % self.frame_skip == 0:
                self._store_log_data(current_time, target_dynamic)

                if self.noise_model:
                    self._compute_noise_emissions()
            

            # Check for final target reached only if all the other waypoints have been reached
            if stop_at_target and self.current_seg_idx == len(self.waypoints):
                final_target = np.array([
                    self.waypoints[-1]['x'], self.waypoints[-1]['y'], self.waypoints[-1]['z']])
                if np.linalg.norm(self.drone.state['pos'] - final_target) < self.target_reached_threshold:
                    self.navigation_time = current_time
                    self.has_reached_target = True
                    if verbose:
                        print(f"Final target reached at time: {current_time:.2f} s")
                    break

            # Check if drone is not moving 
            if current_time > 5 and np.linalg.norm(self.horiz_speed_history) < 1e-2 and np.linalg.norm(self.vertical_speed_history) < 1e-2 and stop_sim_if_not_moving:
                self.navigation_time = current_time
                self.has_moved = False
                if verbose:
                    print(f"Drone stopped at time because was not moving: {current_time:.2f} s")
                    print(f"To change this behavior, set stop_sim_if_not_moving to False.")
                break

        # End timer
        self.simulation_time = time.time() - t_0
        if verbose:
            print(f"Simulation completed in {self.simulation_time:.2f} seconds.")


    def setWind(self, max_simulation_time: float, dt: float, height: float = 100,
                airspeed: float = 10, turbulence_level: int = 30,
                axis=['u', 'v', 'w'], plot_wind_signal: bool = False, seed=None):
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
        """
        num_steps = int(max_simulation_time / dt)
        if isinstance(axis, str):
            axis = [axis]
        for ax in axis:
            self.wind_signals.append(
                dryden_response(axis=ax, height=height, airspeed=airspeed,
                                turbulence_level=turbulence_level, time_steps=num_steps, seed=seed)
            )

        # Plot wind signals if requested
        if plot_wind_signal:
            fig, axs = plt.subplots(len(axis), 1, figsize=(10, 3 * len(axis)))
            for i, ax in enumerate(axis):
                axs[i].plot(np.arange(num_steps) * dt, self.wind_signals[i])
                axs[i].set_title(f"Wind Signal for Axis {ax}")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel("Wind Speed (m/s)")
            plt.tight_layout()
            plt.show()

        self.simulate_wind = True

    def _get_dynamic_target_position(self, seg_start: np.ndarray,
                               seg_end: np.ndarray, v_des: float,
                               k: float = 1.0) -> tuple:
        """
        Compute the dynamic target point along a segment with a look-ahead distance of L = k*v_des.
        Ensures that the target position along the segment does not move backward if the drone regresses.
        Parameters:
            drone_pos (np.ndarray): Current drone position [x, y, z].
            seg_start (np.ndarray): Start point of the segment.
            seg_end (np.ndarray): End point of the segment.
            v_des (float): Desired speed for this segment.
            k (float): Scaling factor for the look-ahead distance.
        Returns:
            tuple: (target, distance) where target is the dynamic target point [x, y, z],
                and distance is the distance from the drone to the next waypoint.
        """
        drone_pos = self.drone.state['pos']
        seg_vector = seg_end - seg_start
        seg_length = np.linalg.norm(seg_vector)

        if seg_length == 0:
            return seg_end, 1.0
        seg_dir = seg_vector / seg_length

        proj_length = np.dot(drone_pos - seg_start, seg_dir)  # Projected length of drone position onto segment
        L = k * v_des  # Look-ahead distance based on desired speed

        # Ensure target does not move backward along the segment
        idx = self.waypoints.index({'x': seg_end[0], 'y': seg_end[1], 'z': seg_end[2], 'v': v_des}) \
            if {'x': seg_end[0], 'y': seg_end[1], 'z': seg_end[2], 'v': v_des} in self.waypoints else None
        if idx is not None:
            self.max_target_length[idx] = max(self.max_target_length[idx], proj_length + L)
            target_length = min(self.max_target_length[idx], seg_length)
        else:
            target_length = min(proj_length + L, seg_length)

        target = seg_start + target_length * seg_dir  # Compute target position along the segment

        # Calculate distance from next waypoint
        distance = np.linalg.norm(drone_pos - seg_end)
        return target, distance

    def _compute_dynamic_target(self, current_seg_idx: int,
                               seg_start: np.ndarray,
                               seg_end: np.ndarray,
                               v_des: float,
                               k_lookahead: float = 1.0) -> tuple:
        """Return the dynamic target and update the current segment if needed.

        Parameters
        current_seg_idx : int
            Index of the current segment.
        seg_start : np.ndarray
            Start point of the current segment.
        seg_end : np.ndarray
            End point of the current segment.
        v_des : float
            Desired speed for the current segment.
        k_lookahead : float, optional
            Scaling factor for the look-ahead distance, by default ``1.0``.

        Returns
        tuple
            ``(target, current_seg_idx, seg_start, seg_end, v_des)`` with the
            new target and updated segment information.
        """
        target, distance = self._get_dynamic_target_position(
            seg_start, seg_end, v_des, k=k_lookahead)

        if distance <= self.target_shift_threshold_distance:
            current_seg_idx += 1
            if current_seg_idx < len(self.waypoints):
                seg_start = seg_end
                seg_end = np.array([
                    self.waypoints[current_seg_idx]['x'],
                    self.waypoints[current_seg_idx]['y'],
                    self.waypoints[current_seg_idx]['z']
                ])
                v_des = self.waypoints[current_seg_idx]['v']
                target, _ = self._get_dynamic_target_position(
                    seg_start, seg_end, v_des, k=k_lookahead)
            else:
                target = seg_end
                current_seg_idx = len(self.waypoints)

        return target, current_seg_idx, seg_start, seg_end, v_des

    def _compute_static_target(self, current_seg_idx: int,
                              seg_end: np.ndarray) -> tuple:
        """Return a static target positioned at the next waypoint.

        The target is simply the next waypoint. When the drone comes within
        ``target_shift_threshold_distance`` of the current waypoint, the target
        is updated to the following one until the final waypoint is reached.

        Parameters
        current_seg_idx : int
            Index of the current waypoint.
        seg_end : np.ndarray
            Current target waypoint.

        Returns
        tuple
            ``(target, current_seg_idx, seg_end)`` with the new target and
            updated waypoint information.
        """
        drone_pos = self.drone.state['pos']
        distance = np.linalg.norm(drone_pos - seg_end)

        if distance <= self.target_shift_threshold_distance:
            current_seg_idx += 1
            if current_seg_idx < len(self.waypoints):
                seg_end = np.array([
                    self.waypoints[current_seg_idx]['x'],
                    self.waypoints[current_seg_idx]['y'],
                    self.waypoints[current_seg_idx]['z']
                ])
            else:
                current_seg_idx = len(self.waypoints)

        return seg_end, current_seg_idx, seg_end
    
    def _clear_histories(self):
        """
        Clear all histories collected during the simulation.
        This method resets all data collections to empty lists.
        """
        self.positions.clear()
        self.angles_history.clear()
        self.rpms_history.clear()
        self.time_history.clear()
        self.horiz_speed_history.clear()
        self.vertical_speed_history.clear()
        self.targets.clear()
        self.power_history.clear()
        self.spl_history.clear()
        self.swl_history.clear()
        self.thrust_history.clear()
        self.delta_b_history.clear()
        self.thrust_no_wind_history.clear()
        self.distance_history.clear()
        self.seg_idx_history.clear()

        self.has_moved = True
        self.has_reached_target = False
        self.navigation_time = None
        self.simulation_time = 0.0
        self.current_seg_idx = 0

        self.noise_emission_map.clear()

    def _store_log_data(self, current_time, target_dynamic):
        """
        Store log data for the current simulation step.
        """
        self.positions.append(self.drone.state['pos'].copy())
        self.angles_history.append(self.drone.state['angles'].copy())
        self.rpms_history.append(self.drone.state['rpm'].copy())
        self.time_history.append(current_time)
        self.horiz_speed_history.append(np.linalg.norm(self.drone.state['vel'][:2]))
        self.vertical_speed_history.append(self.drone.state['vel'][2])
        self.targets.append(target_dynamic.copy())
        self.thrust_history.append(self.drone.thrust)
        self.delta_b_history.append(self.drone.delta_b)
        self.thrust_no_wind_history.append(self.drone.thrust_no_wind)
        self.power_history.append(np.sum(self.drone.state['power']))

        idx = min(self.current_seg_idx, len(self.waypoints)-1)
        wp = self.waypoints[idx]
        dist = np.linalg.norm(self.drone.state['pos'] - np.array([wp['x'], wp['y'], wp['z']]))
        self.distance_history.append(dist)
        self.seg_idx_history.append(self.current_seg_idx)

    def _compute_noise_emissions(self):
        # Compute noise emissions around the drone
        avg_spl = 0.0
        avg_swl = 0.0
        x_d, y_d, z_d = self.drone.state['pos']
        # If position is not valid, set position to zero
        if np.isnan(x_d) or np.isnan(y_d) or np.isnan(z_d):
            x_d, y_d, z_d = 0.0, 0.0, MIN_HEIGHT_FROM_GROUND
        areas, params = self.world.get_areas_in_circle(
            x=int(x_d), y=int(y_d), height=MIN_HEIGHT_FROM_GROUND,
            radius=self.noise_annoyance_radius, include_areas_out_of_bounds=True)
        for area in areas:
            x_a, y_a, _ = area
            dist = np.linalg.norm([x_d, y_d, z_d] - np.array([x_a, y_a, MIN_HEIGHT_FROM_GROUND]))
            zeta = np.arctan2(abs(z_d), dist)
            spl, swl = self.noise_model.get_noise_emissions(
                zeta_angle=zeta, rpms=self.drone.state['rpm'], distance=dist)
            avg_spl += spl
            avg_swl += swl

            if self.generate_sound_emission_map:
                self.noise_emission_map[x_a, y_a] = {
                    'spl': self.noise_emission_map[x_a, y_a]['spl'] + spl if (x_a, y_a) in self.noise_emission_map else spl,
                    'swl': self.noise_emission_map[x_a, y_a]['swl'] + swl if (x_a, y_a) in self.noise_emission_map else swl,
                }

        count = len(areas) if areas else 1
        self.spl_history.append(avg_spl / count)
        self.swl_history.append(avg_swl / count)

