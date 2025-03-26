import numpy as np
from Drone import QuadcopterModel

# --- Main: Simulation and Plotting ---

class Simulation:
    def __init__(self, drone: QuadcopterModel, waypoints: list, dt=0.007, max_simulation_time=200.0, frame_skip=8, target_reached_threshold=2.0, dynamic_target_shift_threshold_prc=0.7):
        self.dt = dt
        self.max_simulation_time = max_simulation_time
        self.num_steps = int(max_simulation_time / dt)
        self.frame_skip = frame_skip
        self.target_reached_threshold = target_reached_threshold
        self.dynamic_target_shift_threshold_prc = dynamic_target_shift_threshold_prc
        self.waypoints = waypoints
        self.drone = drone

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