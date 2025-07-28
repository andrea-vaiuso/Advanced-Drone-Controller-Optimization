import json
import os
from datetime import datetime
from time import time

import numpy as np
from bayes_opt import BayesianOptimization
from World import World
import main as mainfunc

# Optimization parameters
iteration = 0
n_iter = 5000
costs = []
best_target = -float('inf')
best_params = None
simulation_time = 150.0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join('Optimizations', 'Bayesian', timestamp)
os.makedirs(base_dir, exist_ok=True)

opt_output_path = os.path.join(base_dir, 'best_parameters.txt')
log_path = os.path.join(base_dir, 'optimization_log.json')

start_time = time()
last_time = start_time


def log_step(params: dict, cost: float) -> None:
    """Append a single optimization step to the JSON log file."""
    global last_time
    current = time()
    entry = {
        'target': -cost,
        'params': {k: ([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v))
                   for k, v in params.items()},
        'datetime': {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed': current - start_time,
            'delta': current - last_time,
        },
    }
    with open(log_path, 'a') as f:
        json.dump(entry, f)
        f.write('\n')
    last_time = current

parameters = mainfunc.load_parameters("parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()

world = World.load_world(parameters['world_data_path'])


def simulate_pid(pid_gains):
    """Run a simulation with the given PID gains and return the cost metrics."""

    # Initial drone state
    init_state = mainfunc.create_initial_state()

    quad_controller = mainfunc.create_quadcopter_controller(init_state=init_state, pid_gains=pid_gains, t_max=thrust_max, parameters=parameters)
    # Initialize the quadcopter model with the rotor model
    # Default values are taken from the paper: "Modeling of a Quadcopter Trajectory Tracking System Using PID Controller" by Sabir et.al. (2020)
    drone = mainfunc.create_quadcopter_model(init_state=init_state, quad_controller=quad_controller, parameters=parameters)

    # Initialize the simulation
    sim = mainfunc.create_simulation(drone=drone, world=world, waypoints=waypoints, parameters=parameters, noise_model=None)

    # sim.setWind(max_simulation_time=simulation_time, dt=dt, height=100, airspeed=10, turbulence_level=10, plot_wind_signal=False, seed = None)
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True)

    # Collect results
    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time
    
    final_target = {'x': waypoints[-1]['x'], 'y': waypoints[-1]['y'], 'z': waypoints[-1]['z']}
    final_distance = np.linalg.norm(sim.drone.state['pos'] - np.array([final_target['x'],
                                                              final_target['y'],
                                                              final_target['z']]))

    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0]))) # Pitch oscillation calculated as the sum of absolute differences in pitch angles
    roll_osc  = np.sum(np.abs(np.diff(angles[:, 1]))) # Roll oscillation calculated as the sum of absolute differences in roll angles
    thrust_osc = np.sum(np.abs(np.diff(sim.thrust_history))) * 1e-5 # Thrust oscillation calculated as the sum of absolute differences in thrust values
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc + thrust_osc)
    if not sim.has_moved: cost += 1000
    elif not sim.has_reached_target: cost += 1000

    return cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, sim.has_reached_target

def objective(kp_pos, ki_pos, kd_pos,
              kp_alt, ki_alt, kd_alt,
              kp_att, ki_att, kd_att,
              kp_hsp, ki_hsp, kd_hsp,
              kp_vsp, ki_vsp, kd_vsp):
    """Objective function called by the Bayesian optimizer."""
    global iteration, best_target, best_params

    iteration += 1
    params = {
        'k_pid_pos': (kp_pos, ki_pos, kd_pos),
        'k_pid_alt': (kp_alt, ki_alt, kd_alt),
        'k_pid_att': (kp_att, ki_att, kd_att),
        'k_pid_yaw': (0.5, 1e-6, 0.1),
        'k_pid_hsp': (kp_hsp, ki_hsp, kd_hsp),
        'k_pid_vsp': (kp_vsp, ki_vsp, kd_vsp),
    }
    cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, targ_reached = simulate_pid(
        pid_gains=params
    )
    log_step(params, cost)
    target = -cost  # bayes_opt maximizes

    # If this target is better than the best so far, update the file
    if target > best_target:
        best_target = target
        best_params = {
            'k_pid_pos': (kp_pos, ki_pos, kd_pos),
            'k_pid_alt': (kp_alt, ki_alt, kd_alt),
            'k_pid_att': (kp_att, ki_att, kd_att),
            'k_pid_yaw': (0.5, 1e-6, 0.1),
            'k_pid_hsp': (kp_hsp, ki_hsp, kd_hsp),
            'k_pid_vsp': (kp_vsp, ki_vsp, kd_vsp),
            'final_time': final_time,
            'cost': cost,
        }
        # write to file
        with open("opt_temp.txt", 'w') as f:
            f.write(f"Iteration: {iteration}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"target = {best_target}\n")

    print(f"{iteration}/{n_iter}: cost={cost:.4f}, best target={best_target:.4f}, final_time={final_time:.2f}s, final_distance={final_distance:.2f}m, pitch_osc={pitch_osc:.2f}, roll_osc={roll_osc:.2f}, thrust_osc={thrust_osc:.2f}, target_reached={targ_reached}")
    return target

def seconds_to_hhmmss(seconds):
    """Convert seconds to a ``HH:MM:SS`` formatted string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def main():
    """Run the Bayesian PID gain optimization."""
    # Define the bounds for the optimization variables
    pbounds = {
        'kp_pos': (1e-6, 1e3), 
        'ki_pos': (1e-6, 1e3), 
        'kd_pos': (1e-6, 1e3),

        'kp_alt': (1e-6, 1e3),   
        'ki_alt': (1e-6, 1e3),  
        'kd_alt': (1e-6, 1e3),

        'kp_att': (1e-6, 1e3),     
        'ki_att': (1e-6, 1e3),  
        'kd_att': (1e-6, 1e3),

        'kp_hsp': (1e-6, 1e3),
        'ki_hsp': (1e-6, 1e3),
        'kd_hsp': (1e-6, 1e3),

        'kp_vsp': (1e-6, 1e3),
        'ki_vsp': (1e-6, 1e3),
        'kd_vsp': (1e-6, 1e3)
    }
    # init_guess = {
    #     'kp_pos': 0.7605314210227943,
    #     'ki_pos': 0.37624518297791576,
    #     'kd_pos': 1.0,

    #     'kp_alt': 196.60373623182426,
    #     'ki_alt': 29.090249051600722,
    #     'kd_alt': 107.90640578767405,

    #     'kp_att': 2.3755182014455283,
    #     'ki_att': 1e-05,
    #     'kd_att': 2.808073996317681,

    #     'kp_hsp': 124.37310350657175,
    #     'ki_hsp': 1e-05,
    #     'kd_hsp': 0.7454357201860323,

    #     'kp_vsp': 114.68965027417173,
    #     'ki_vsp': 1.0,
    #     'kd_vsp': 0.6251197244454744
    # }

    start_time = time()
    print("Starting optimization...")

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )

    # optimizer.probe(
    #     params=init_guess,
    #     lazy=True,
    # )

    optimizer.maximize(
        init_points=50,
        n_iter=n_iter,
    )
    tot_time = time() - start_time
    print(f"Optimization completed in {tot_time:.2f} seconds.")
    print("Best parameters found:")
    best = optimizer.max['params']
    best_formatted = {
        'k_pid_pos': (best['kp_pos'], best['ki_pos'], best['kd_pos']),
        'k_pid_alt': (best['kp_alt'], best['ki_alt'], best['kd_alt']),
        'k_pid_att': (best['kp_att'], best['ki_att'], best['kd_att']),
        'k_pid_yaw': (0.5, 1e-6, 0.1),
        'k_pid_hsp': (best['kp_hsp'], best['ki_hsp'], best['kd_hsp']),
        'k_pid_vsp': (best['kp_vsp'], best['ki_vsp'], best['kd_vsp']),
    }

    print("k_pid_yaw = (0.5, 1e-6, 0.1)")
    print("k_pid_pos = ({:.5g}, {:.5g}, {:.5g})".format(*best_formatted['k_pid_pos']))
    print("k_pid_alt = ({:.5g}, {:.5g}, {:.5g})".format(*best_formatted['k_pid_alt']))
    print("k_pid_att = ({:.5g}, {:.5g}, {:.5g})".format(*best_formatted['k_pid_att']))
    print("k_pid_hsp = ({:.5g}, {:.5g}, {:.5g})".format(*best_formatted['k_pid_hsp']))
    print("k_pid_vsp = ({:.5g}, {:.5g}, {:.5g})".format(*best_formatted['k_pid_vsp']))
    print("Best target value: {:.4f}".format(optimizer.max['target']))

    with open(opt_output_path, 'w') as f:
        f.write("Best parameters found:\n")
        for k, v in best_formatted.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Best target value: {optimizer.max['target']}\n")
        f.write(f"Total optimization time: {seconds_to_hhmmss(tot_time)}\n")
        f.write(f"N iterations: {n_iter}\n")
        f.write(f"Max sim time: {simulation_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
