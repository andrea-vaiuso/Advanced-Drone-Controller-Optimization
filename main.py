
import numpy as np
from Drone import QuadcopterModel
from Controller import QuadCopterController
from Simulation import Simulation
from plotting_functions import plot3DAnimation, plotLogData

def main():
    """
    Run the drone simulation and plot the results.
    The simulation stops early if the drone reaches the final target (within a 2-meter threshold).
    """
    # Simulation parameters
    dt = 0.007
    simulation_time = 200.0
    frame_skip = 8
    threshold = 2.0  # Stop simulation if within 2 meters of final target
    dynamic_target_shift_threshold_prc = 0.7 # Shift to next segment if a certain percentage of current segment is covered

    # --- Define Waypoints (with desired speed) ---
    waypoints = [
        {'x': 10.0, 'y': 10.0, 'z': 70.0, 'v': 10},  # Start near origin at high altitude
        {'x': 90.0, 'y': 10.0, 'z': 70.0, 'v': 10},  # Far in x, near y, maintaining high altitude
        {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v': 0.5},   # Far in both x and y with even higher altitude
        {'x': 10.0, 'y': 90.0, 'z': 20.0, 'v': 10},   # Sharp maneuver: near x, far y with dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0, 'v': 10},   # Central target with intermediate altitude
        {'x': 60.0, 'y': 60.0, 'z': 40.0, 'v': 10},   # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 40.0, 'v': 10},   # Hovering target 2
        {'x': 80.0, 'y': 80.0, 'z': 40.0, 'v': 10},   # Hovering target 3
        {'x': 10.0, 'y': 10.0, 'z': 10.0, 'v': 10}    # Final target: near origin at low altitude
    ]

    # Initial drone state
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0])
    }
    start_position = state['pos'].copy()

    # PID controller settings (yaw gains remain fixed)
    kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1
    kp_pos, ki_pos, kd_pos = 0.080397, 6.6749e-07, 0.18084
    kp_alt, ki_alt, kd_alt = 6.4593, 0.00042035, 10.365
    kp_att, ki_att, kd_att = 2.7805, 0.00045168, 0.36006

    # Drone physical parameters
    params = {
        'm': 5.2,
        'g': 9.81,
        'I': np.array([3.8e-3, 3.8e-3, 7.1e-3]),
        'b': 3.13e-5,
        'd': 7.5e-7,
        'l': 0.32,
        'Cd': np.array([0.1, 0.1, 0.15]),
        'Ca': np.array([0.1, 0.1, 0.15]),
        'Jr': 6e-5
    }

    # Initialize the quadcopter controller and model
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     
        kp_alt, ki_alt, kd_alt,     
        kp_att, ki_att, kd_att,
        kp_yaw, ki_yaw, kd_yaw,   
        m=params['m'], g=params['g'], b=params['b'],
        u1_limit=100.0, u2_limit=10.0, u3_limit=5.0, u4_limit=10.0
    )

    drone = QuadcopterModel(
        m=params['m'],
        I=params['I'],
        b=params['b'],
        d=params['d'],
        l=params['l'],
        Cd=params['Cd'],
        Ca=params['Ca'],
        Jr=params['Jr'],
        init_state=state,
        controller=quad_controller,
        max_rpm=10000.0
    )

    # Initialize the simulation
    sim = Simulation(drone, 
                    waypoints, 
                    dt=dt,
                    max_simulation_time=simulation_time,
                    frame_skip=frame_skip, 
                    target_reached_threshold=threshold, 
                    dynamic_target_shift_threshold_prc=dynamic_target_shift_threshold_prc)
    
    positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, targets = sim.startSimulation()

    # Plot 3D animation of the drone's trajectory and attitude
    plot3DAnimation(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, targets, waypoints, start_position, dt, frame_skip)
    plotLogData(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, waypoints)

   
if __name__ == "__main__":
    main()
