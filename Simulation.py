# Author: Andrea Vaiuso
# Version: 2.0
# Date: 15.07.2025
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
                 dynamic_target_shift_threshold_distance: float = 5,
                 noise_model: RotorSoundModel = None, noise_annoyance_radius: int = 100):
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
            dynamic_target_shift_threshold_distance (float): Distance to consider for shifting the target.
            noise_model (RotorSoundModel): Optional noise model for simulating drone noise emissions.
            noise_annoyance_radius (int): Radius around the drone to consider for noise emissions.

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
        self.dynamic_target_shift_threshold_distance = dynamic_target_shift_threshold_distance
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
        self.delta_b_history = []
        self.thrust_no_wind_history = []

        # Simulation runtime
        self.simulation_time = 0.0
        self.navigation_time = None
        self.has_moved = True
        self.has_reached_target = False

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

    def _compute_moving_target(self, drone_pos: np.ndarray, seg_start: np.ndarray,
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
                and distance is the distance from the drone to the target.
        """
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

        # Calculate distance from target
        distance = np.linalg.norm(drone_pos - target)
        return target, distance

    def startSimulation(self, stop_at_target: bool = True, verbose: bool = True, stop_sim_if_not_moving: bool = False):
        """
        Start the simulation of the drone following dynamic targets along the waypoints.
        Parameters:
            stop_at_target (bool): If True, stop when the final target is reached.
            verbose (bool): If True, print simulation progress and completion messages.
            stop_sim_if_not_moving (bool): If True, stop simulation if the drone is not moving for a certain period.

        This method updates the drone's state at each time step and stores data in class attributes.
        """
        # Reset runtime and histories
        self.simulation_time = 0.0
        self.positions.clear()
        self.angles_history.clear()
        self.rpms_history.clear()
        self.time_history.clear()
        self.horiz_speed_history.clear()
        self.vertical_speed_history.clear()
        self.targets.clear()
        self.spl_history.clear()
        self.swl_history.clear()
        self.thrust_history.clear()
        self.delta_b_history.clear()
        self.thrust_no_wind_history.clear()

        # Reset drone state to initial conditions
        self.drone.reset_state() 

        # Initialize histories for dynamic targeting
        self.max_target_length = [0.0] * len(self.waypoints)

        # Start timer
        t_0 = time.time()

        # Initialize dynamic targeting
        current_seg_idx = 0
        seg_start = self.drone.state['pos'].copy()
        seg_end = np.array([self.waypoints[0]['x'], self.waypoints[0]['y'], self.waypoints[0]['z']])
        v_des = self.waypoints[0]['v']
        k_lookahead = 1.0

        num_steps = int(self.max_simulation_time / self.dt)

        for step in range(num_steps):
            # Compute dynamic target
            target_dynamic, distance = self._compute_moving_target(
                self.drone.state['pos'], seg_start, seg_end, v_des, k=k_lookahead)

            # Shift to next segment if needed
            if distance <= self.dynamic_target_shift_threshold_distance:
                current_seg_idx += 1
                if current_seg_idx < len(self.waypoints):
                    seg_start = seg_end
                    seg_end = np.array([
                        self.waypoints[current_seg_idx]['x'],
                        self.waypoints[current_seg_idx]['y'],
                        self.waypoints[current_seg_idx]['z']])
                    v_des = self.waypoints[current_seg_idx]['v']
                    target_dynamic, _ = self._compute_moving_target(
                        self.drone.state['pos'], seg_start, seg_end, v_des, k=k_lookahead)
                else:
                    target_dynamic = seg_end
                    current_seg_idx = len(self.waypoints)  # Final segment reached

            # Update drone state
            self.drone.update_state({'x': target_dynamic[0], 'y': target_dynamic[1], 'z': target_dynamic[2]},
                                     self.dt, verbose=False)
            # Apply wind if enabled
            if self.simulate_wind and len(self.wind_signals) >= 3:
                self.drone.update_wind(self.wind_signals[2][step], simulate_wind=True) #Only use 'w' axis wind for vertical component

            current_time = step * self.dt

            # Store data at specified intervals
            if step % self.frame_skip == 0:
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

                if self.noise_model:
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
                    count = len(areas) if areas else 1
                    self.spl_history.append(avg_spl / count)
                    self.swl_history.append(avg_swl / count)
            # Check for final target reached only if all the other waypoints have been reached
            if stop_at_target and current_seg_idx == len(self.waypoints):
                final_target = np.array([
                    self.waypoints[-1]['x'], self.waypoints[-1]['y'], self.waypoints[-1]['z']])
                if np.linalg.norm(self.drone.state['pos'] - final_target) < self.target_reached_threshold:
                    if verbose:
                        print(f"Final target reached at time: {current_time:.2f} s")
                    self.navigation_time = current_time
                    self.has_reached_target = True
                    break

            # Check if drone is not moving 
            if current_time > 5 and np.linalg.norm(self.horiz_speed_history) < 1e-2 and np.linalg.norm(self.vertical_speed_history) < 1e-2 and stop_sim_if_not_moving:
                
                if verbose:
                    print(f"Drone stopped at time because was not moving: {current_time:.2f} s")
                    print(f"To change this behavior, set stop_sim_if_not_moving to False.")
                self.navigation_time = current_time
                self.has_moved = False
                break

        # End timer
        self.simulation_time = time.time() - t_0
        if verbose:
            print(f"Simulation completed in {self.simulation_time:.2f} seconds.")
