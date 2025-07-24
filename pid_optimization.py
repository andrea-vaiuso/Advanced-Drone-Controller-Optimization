import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs

from Drone import QuadcopterModel
from Controller import QuadCopterController
from Simulation import Simulation
from World import World
from Noise.DNNModel import RotorSoundModel as DNNModel
from Noise.EmpaModel import NoiseModel as EmpaModel
from Rotor.TorchRotorModel import RotorModel

import os
import json

# Optimization parameters
iteration = 0
n_iter = 200
costs = []
best_target = -float('inf')
best_params = None
best_file = 'best_pid.txt'

# Simulation parameters
dt = 0.007
simulation_time = 150 # seconds
frame_skip = 16 # Number of frames to skip for smoother animation
threshold = 2.0  # Stop simulation if within 2 meters of final target
dynamic_target_shift_threshold_distance = 3 # Shift to next segment if a certain percentage of current segment is covered

max_rpm = 3000 # Maximum RPM for the motors
n_rotors = 4 # Number of rotors
max_angle_deg = 30 # Maximum tilt angle in degrees
max_angle_rad = np.radians(max_angle_deg) # Convert to radians

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

demo_rotor_model = RotorModel(norm_params_path="Rotor/normalization_params.pth")
demo_rotor_model.load_model('Rotor/rotor_model.pth')
t_max, _, _, _, _, _ = demo_rotor_model.predict_aerodynamic(max_rpm)

# PID controller settings (yaw remain fixed)
kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1

print(f"Settings Controller limits: u1={t_max*n_rotors}, u2={max_angle_rad}, u3={max_angle_rad}, u4={max_angle_rad}")
# Initialize the quadcopter controller and model

def simulate_pid(kp_pos, ki_pos, kd_pos,
                 kp_alt, ki_alt, kd_alt,
                 kp_att, ki_att, kd_att):

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

    # Initialize the simulation
    sim = Simulation(drone,
                    world,
                    waypoints, 
                    dt=dt,
                    max_simulation_time=simulation_time,
                    frame_skip=frame_skip,
                    target_reached_threshold=threshold,
                    dynamic_target_shift_threshold_distance=dynamic_target_shift_threshold_distance,
                    noise_model=None,
                    )

    # sim.setWind(max_simulation_time=simulation_time, dt=dt, height=100, airspeed=10, turbulence_level=10, plot_wind_signal=False, seed = None)
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True)

    # Collect results
    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time * 3
    
    final_target = {'x': waypoints[-1]['x'], 'y': waypoints[-1]['y'], 'z': waypoints[-1]['z']}
    final_distance = np.linalg.norm(sim.drone.state['pos'] - np.array([final_target['x'],
                                                              final_target['y'],
                                                              final_target['z']]))

    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0])))
    roll_osc  = np.sum(np.abs(np.diff(angles[:, 1])))
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc)
    if sim.drone_didnt_move: cost += 1000
    
    return cost, final_time, final_distance, pitch_osc, roll_osc

def objective(kp_pos, ki_pos, kd_pos,
              kp_alt, ki_alt, kd_alt,
              kp_att, ki_att, kd_att):
    global iteration, best_target, best_params

    iteration += 1
    cost, final_time, final_distance, pitch_osc, roll_osc = simulate_pid(
        kp_pos, ki_pos, kd_pos,
        kp_alt, ki_alt, kd_alt,
        kp_att, ki_att, kd_att
    )
    target = -cost  # bayes_opt massimizza

    # se questo target è migliore del best-so-far, aggiorna file
    if target > best_target:
        best_target = target
        best_params = {
            'kp_pos': kp_pos, 'ki_pos': ki_pos, 'kd_pos': kd_pos,
            'kp_alt': kp_alt, 'ki_alt': ki_alt, 'kd_alt': kd_alt,
            'kp_att': kp_att, 'ki_att': ki_att, 'kd_att': kd_att,
            'final_time': final_time,
            'cost': cost
        }
        # scrivi su file
        with open(best_file, 'w') as f:
            f.write(f"Iteration: {iteration}\n")
            for k, v in best_params.items():
                f.write(f"{k} = {v}\n")
            f.write(f"target = {best_target}\n")

    print(f"{iteration}/{n_iter}: cost={cost:.4f}, best target={best_target:.4f}, final_time={final_time:.2f}s, final_distance={final_distance:.2f}m, pitch_osc={pitch_osc:.2f}, roll_osc={roll_osc:.2f}")
    return target


def main():
    # Define the bounds for the optimization variables
    pbounds = {
        'kp_pos': (1e-7, 1), 
        'ki_pos': (1e-7, 1), 
        'kd_pos': (1e-7, 1),

        'kp_alt': (1e-7, 1),   
        'ki_alt': (1e-7, 1),  
        'kd_alt': (1e-7, 1),

        'kp_att': (1, 200),     
        'ki_att': (1e-7, 1),  
        'kd_att': (1, 200)
    }
    log_path = "bayes_opt_logs.json"

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )
    logger = JSONLogger(path=log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    if os.path.exists(log_path):
        load_logs(optimizer, logs=[log_path])
        with open(log_path) as f:
            past = json.load(f)
        done = len(past)
        print(f"✅ Resumed from {done} completed iterations.")
    else:
        done = 0
        print("🚀 Starting from scratch.")

    # 5) Calculate how many are left
    remaining = max(n_iter - done, 0)
    if remaining == 0:
        print("All iterations were already completed!")
        return

    # 6) Start the optimization:
    # - init_points=5 only if done==0 (otherwise 0)
    optimizer.maximize(
        init_points=5 if done == 0 else 0,
        n_iter=remaining,
    )
    
    print("Best parameters found:")
    best = optimizer.max['params']

    print("kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1")
    print("kp_pos, ki_pos, kd_pos = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_pos'], best['ki_pos'], best['kd_pos']))
    print("kp_alt, ki_alt, kd_alt = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_alt'], best['ki_alt'], best['kd_alt']))
    print("kp_att, ki_att, kd_att = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_att'], best['ki_att'], best['kd_att']))

if __name__ == "__main__":
    main()