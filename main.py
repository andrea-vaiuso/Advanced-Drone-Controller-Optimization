import numpy as np
from Drone import QuadcopterModel
from Controller import QuadCopterController
from Simulation import Simulation
from plotting_functions import plot3DAnimation, plotLogData
from World import World
from Noise.DNNModel import RotorSoundModel as DNNModel
from Noise.EmpaModel import NoiseModel as EmpaModel
from Rotor.TorchRotorModel import RotorModel

def main():
    """
    Run the drone simulation and plot the results.
    """
    # Simulation parameters
    dt = 0.007
    simulation_time = 35 # seconds
    frame_skip = 16 # Number of frames to skip for smoother animation
    threshold = 2.0  # Stop simulation if within 2 meters of final target
    dynamic_target_shift_threshold_distance = 3 # Shift to next segment if within 3 meters of current segment



    max_rpm = 3000 # Maximum RPM for the motors
    n_rotors = 4 # Number of rotors
    max_angle_deg = 30 # Maximum tilt angle in degrees
    max_angle_rad = np.radians(max_angle_deg) # Convert to radians

    # # --- Define Waypoints (with desired speed) ---
    waypoints = [
        {'x': 10.0, 'y': 10.0, 'z': 70.0, 'v': 5},  # Start near origin at high altitude
        {'x': 90.0, 'y': 10.0, 'z': 70.0, 'v': 5},  # Far in x, near y, maintaining high altitude
        {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v': 5},   # Far in both x and y with even higher altitude
        {'x': 10.0, 'y': 90.0, 'z': 20.0, 'v': 5},   # Sharp maneuver: near x, far y with dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0, 'v': 5},   # Central target with intermediate altitude
        {'x': 60.0, 'y': 60.0, 'z': 40.0, 'v': 5},   # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 40.0, 'v': 5},   # Hovering target 2
        {'x': 80.0, 'y': 80.0, 'z': 40.0, 'v': 5},   # Hovering target 3
        {'x': 10.0, 'y': 10.0, 'z': 10.0, 'v': 5}    # Final target: near origin at low altitude
    ]

    # waypoints = [
    #     {'x': 30.0, 'y': 30.0, 'z': 90.0, 'v': 4},
    #     {'x': 30.0, 'y': 30.0, 'z': 50.0, 'v': 4},
    #     # {'x': 10.0, 'y': 10.0, 'z': 10.0, 'v': 5},
    #     # {'x': 20.0, 'y': 20.0, 'z': 10.0, 'v': 5},
    #     {'x': 30.0, 'y': 30.0, 'z': 10.0, 'v': 4},
    #     # {'x': 40.0, 'y': 40.0, 'z': 90.0, 'v': 5},
    #     # {'x': 50.0, 'y': 50.0, 'z': 10.0, 'v': 5},
    #     # {'x': 60.0, 'y': 60.0, 'z': 10.0, 'v': 5},
    #     # {'x': 70.0, 'y': 70.0, 'z': 10.0, 'v': 5},
    #     # {'x': 80.0, 'y': 80.0, 'z': 10.0, 'v': 5},
    #     {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v': 4}
    # ]

    # Initial drone state
    state = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]), # roll, pitch, yaw # check if is consistent with the coordinate system
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0]),
        'thrust': 0.0,  # Initial thrust
        'torque': 0.0, # Initial torque
        'power': 0.0, # Initial power
    }
    start_position = state['pos'].copy()

    demo_rotor_model = RotorModel(norm_params_path="Rotor/normalization_params.pth")
    demo_rotor_model.load_model('Rotor/rotor_model.pth')
    t_max, _, _, _, _, _ = demo_rotor_model.predict_aerodynamic(max_rpm)

    # PID controller settings (yaw remain fixed)
    kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1
    kp_pos, ki_pos, kd_pos = 0.7, 0.1, 5.25
    kp_alt, ki_alt, kd_alt = 150.1, 20, 500.1
    kp_att, ki_att, kd_att = 1.7, 1e-4, 1.25

    print(f"Settings Controller limits: u1={t_max*n_rotors}, u2={max_angle_rad}, u3={max_angle_rad}, u4={max_angle_rad}")
    # Initialize the quadcopter controller and model
    quad_controller = QuadCopterController(
        state, 
        kp_pos, ki_pos, kd_pos,     
        kp_alt, ki_alt, kd_alt,     
        kp_att, ki_att, kd_att,
        kp_yaw, ki_yaw, kd_yaw,   
        u1_limit=t_max*n_rotors, u2_limit=max_angle_rad, u3_limit=max_angle_rad, u4_limit=max_angle_rad
    )

    # Initialize the quadcopter model with the rotor model
    # Default values are taken from the paper: "Modeling of a Quadcopter Trajectory Tracking System Using PID Controller" by Sabir et.al. (2020)
    drone = QuadcopterModel(
        m=5.2,
        I=[3.8e-3, 3.8e-3, 7.1e-3],
        d=7.5e-7,
        l=0.32,
        Cd=np.array([0.1, 0.1, 0.15]),
        Ca=np.array([0.1, 0.1, 0.15]),
        Jr=6e-5,
        init_state=state,
        controller=quad_controller,
        n_rotors=n_rotors,  # Number of rotors
        rotor_model_path='Rotor/rotor_model.pth',  # Path to the pre-trained rotor model
        max_rpm=max_rpm,  # Maximum RPM for the motors
    )

    # Initialize the world
    world = World.load_world("Worlds/world_winterthur.pkl")

    noise_model_dnn = DNNModel(
        rpm_reference=2500,
        filename="Noise/angles_swl.npy"
    )

    noise_model_empa = EmpaModel(
        scaler_filename="Noise/scaler.joblib"
    )

    noise_model_empa.load_model("Noise/model_coefficients.npz")

    # Choose the noise model to use
    noise_model = noise_model_dnn  # Use DNN model
    # noise_model = noise_model_empa # Use Empa model

    # Initialize the simulation
    sim = Simulation(drone,
                    world,
                    waypoints, 
                    dt=dt,
                    max_simulation_time=simulation_time,
                    frame_skip=frame_skip,
                    target_reached_threshold=threshold,
                    dynamic_target_shift_threshold_distance=dynamic_target_shift_threshold_distance,
                    noise_model=noise_model)

    # sim.setWind(max_simulation_time=simulation_time, dt=dt, height=100, airspeed=10, turbulence_level=10, plot_wind_signal=False, seed = None)
    sim.startSimulation(stop_at_target=False)

    # Plot 3D animation of the drone's trajectory
    plot3DAnimation(np.array(sim.positions), 
                    np.array(sim.angles_history), 
                    np.array(sim.rpms_history), 
                    np.array(sim.time_history), 
                    np.array(sim.horiz_speed_history), 
                    np.array(sim.vertical_speed_history), 
                    np.array(sim.targets), 
                    waypoints, 
                    start_position, 
                    dt, 
                    frame_skip)
    
    log_dict = {
        'X Position': {
            'data': np.array(sim.positions)[:, 0],
            'ylabel': 'X (m)',
            'color': 'tab:blue',
            'linestyle': '-',
            'label': 'X Position',
            'showgrid': True
        },
        'Y Position': {
            'data': np.array(sim.positions)[:, 1],
            'ylabel': 'Y (m)',
            'color': 'tab:orange',
            'linestyle': '-',
            'label': 'Y Position',
            'showgrid': True
        },
        'Z Position': {
            'data': np.array(sim.positions)[:, 2],
            'ylabel': 'Z (m)',
            'color': 'tab:green',
            'linestyle': '-',
            'label': 'Z Position',
            'showgrid': True
        },
        'Attitude (Pitch, Roll, Yaw)': {
            'data': {
                'Pitch': np.array(sim.angles_history)[:, 0],
                'Roll':  np.array(sim.angles_history)[:, 1],
                'Yaw':   np.array(sim.angles_history)[:, 2]
            },
            'ylabel': 'Angle (rad)',
            'styles': {
                'Pitch': {'color': 'tab:blue',   'linestyle': '-',  'label': 'Pitch'},
                'Roll':  {'color': 'tab:orange', 'linestyle': '--', 'label': 'Roll'},
                'Yaw':   {'color': 'tab:green',  'linestyle': '-.', 'label': 'Yaw'}
            }
        },
        'Motor RPMs': {
            'data': np.array(sim.rpms_history),
            'ylabel': 'RPM',
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
            'linestyles': ['-', '--', '-.', ':'],
            'labels': ['RPM1', 'RPM2', 'RPM3', 'RPM4']
        },
        'Speeds': {
            'data': {
                'Horizontal Speed': np.array(sim.horiz_speed_history),
                'Vertical Speed':   np.array(sim.vertical_speed_history)
            },
            'ylabel': 'Speed (m/s)',
            'styles': {
                'Horizontal Speed': {'color': 'r', 'linestyle': '-', 'label': 'Horizontal Speed'},
                'Vertical Speed':   {'color': 'g', 'linestyle': '-', 'label': 'Vertical Speed'}
            }
        },
        'Sound Pressure Level (SPL)': {
            'data': np.array(sim.spl_history),
            'ylabel': 'Level (dB)',
            'color': 'orange',
            'linestyle': '-',
            'label': 'SPL',
            'showgrid': True
        },
        'Sound Power Level (SWL)': {
            'data': np.array(sim.swl_history),
            'ylabel': 'Level (dB)',
            'color': 'purple',
            'linestyle': '-',
            'label': 'SWL',
            'showgrid': True
        },

    }
    plotLogData(
        log_dict,
        time = np.array(sim.time_history),
        waypoints = waypoints,
        ncols = 3,
    )

if __name__ == "__main__":
    main()
