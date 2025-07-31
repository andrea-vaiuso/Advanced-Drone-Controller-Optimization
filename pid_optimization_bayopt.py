import json
import os
from datetime import datetime
from time import time
import yaml

import numpy as np
from bayes_opt import BayesianOptimization
from World import World
import main as mainfunc
from Simulation import Simulation

from plotting_functions import plot3DAnimation
import opt_func
from opt_func import log_step, calculate_costs, seconds_to_hhmmss, plot_costs_trend

# Optimization parameters loaded from YAML
with open(os.path.join('Settings', 'bay_opt.yaml'), 'r') as f:
    bayopt_cfg = yaml.safe_load(f)

iteration = 0
n_iter = int(bayopt_cfg.get('n_iter', 1500))
costs = []
best_target = -float('inf')
best_params = None
simulation_time = float(bayopt_cfg.get('simulation_time', 150))
init_points = int(bayopt_cfg.get('init_points', 20))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join('Optimizations', 'Bayesian', timestamp)
os.makedirs(base_dir, exist_ok=True)

opt_output_path = os.path.join(base_dir, 'best_parameters.txt')
log_path = os.path.join(base_dir, 'optimization_log.json')

# initialize logging timers in opt_func
opt_func.start_time = time()
opt_func.last_time = opt_func.start_time

parameters = mainfunc.load_parameters("Settings/simulation_parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()

world = World.load_world(parameters['world_data_path'])

noise_model = mainfunc.load_dnn_noise_model(parameters)



def simulate_pid(pid_gains):
    """Run a simulation with the given PID gains and return the cost metrics."""

    # Initial drone state
    init_state = mainfunc.create_initial_state()

    quad_controller = mainfunc.create_quadcopter_controller(init_state=init_state, pid_gains=pid_gains, t_max=thrust_max, parameters=parameters)
    # Initialize the quadcopter model with the rotor model
    # Default values are taken from the paper: "Modeling of a Quadcopter Trajectory Tracking System Using PID Controller" by Sabir et.al. (2020)
    drone = mainfunc.create_quadcopter_model(init_state=init_state, quad_controller=quad_controller, parameters=parameters)

    # Initialize the simulation
    sim = mainfunc.create_simulation(drone=drone, world=world, waypoints=waypoints, parameters=parameters, noise_model=noise_model, generate_sound_map=False)

    # sim.setWind(max_simulation_time=simulation_time, dt=dt, height=100, airspeed=10, turbulence_level=10, plot_wind_signal=False, seed = None)
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True, use_static_target=True)

    # FOR DEBUG PURPOSES plot3DAnimation(np.array(sim.positions), 
                    # np.array(sim.angles_history), 
                    # np.array(sim.rpms_history), 
                    # np.array(sim.time_history), 
                    # np.array(sim.horiz_speed_history), 
                    # np.array(sim.vertical_speed_history), 
                    # np.array(sim.targets), 
                    # waypoints, 
                    # init_state['pos'], 
                    # float(parameters['dt']), 
                    # int(parameters['frame_skip']))

    return calculate_costs(sim, simulation_time)

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
    sim_costs = simulate_pid(pid_gains=params)

    total_cost = sim_costs['total_cost']
    time_cost = sim_costs['time_cost']
    final_distance_cost = sim_costs['final_distance_cost']
    oscillation_cost = sim_costs['oscillation_cost']
    completition_cost = sim_costs['completition_cost']
    overshoot_cost = sim_costs['overshoot_cost']
    power_cost = sim_costs['power_cost']
    noise_cost = sim_costs['noise_cost']
    n_waypoints_completed = sim_costs['n_waypoints_completed']
    tot_waypoints = sim_costs['tot_waypoints']

    log_step(params, total_cost, log_path)
    target = -total_cost  # bayes_opt maximizes

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
            'final_time': time_cost,
            'cost': total_cost,
        }
        # write to file
        with open("opt_temp.txt", 'w') as f:
            f.write(f"Iteration: {iteration}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"target = {best_target}\n")

    print(f"{iteration}/{n_iter}: cost={total_cost:.4f}, best_cost={best_target:.4f}, time_cost={time_cost:.2f}, " + \
           f"distance_cost={final_distance_cost:.2f}, oscillation_cost={oscillation_cost:.2f}, completition_cost={completition_cost:.2f}, overshoot_cost={overshoot_cost:.2f}, power_cost={power_cost:.2f}, noise_cost={noise_cost:.2f} | completed_targets={n_waypoints_completed}/{tot_waypoints}")
    costs.append(total_cost)
    return target


def main():
    """Run the Bayesian PID gain optimization."""
    # Load search bounds from configuration
    pbounds_cfg = bayopt_cfg.get('pbounds', {})
    pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}

    current_best_pid_gains = mainfunc.load_pid_gains(parameters)
    init_guess = {
        'kp_pos': current_best_pid_gains['k_pid_pos'][0],
        'ki_pos': current_best_pid_gains['k_pid_pos'][1],
        'kd_pos': current_best_pid_gains['k_pid_pos'][2],

        'kp_alt': current_best_pid_gains['k_pid_alt'][0],
        'ki_alt': current_best_pid_gains['k_pid_alt'][1],
        'kd_alt': current_best_pid_gains['k_pid_alt'][2],

        'kp_att': current_best_pid_gains['k_pid_att'][0],
        'ki_att': current_best_pid_gains['k_pid_att'][1],
        'kd_att': current_best_pid_gains['k_pid_att'][2],

        'kp_hsp': current_best_pid_gains['k_pid_hsp'][0],
        'ki_hsp': current_best_pid_gains['k_pid_hsp'][1],
        'kd_hsp': current_best_pid_gains['k_pid_hsp'][2],

        'kp_vsp': current_best_pid_gains['k_pid_vsp'][0],
        'ki_vsp': current_best_pid_gains['k_pid_vsp'][1],
        'kd_vsp': current_best_pid_gains['k_pid_vsp'][2]
    }

    start_time = time()
    print("Starting optimization...")
    try:
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )

        optimizer.probe(
            params=init_guess,
            lazy=True,
        )

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        tot_time = time() - start_time
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

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

    print("k_pid_yaw: (0.5, 1e-6, 0.1)")
    print("k_pid_pos: [{:.5g}, {:.5g}, {:.5g}]".format(*best_formatted['k_pid_pos']))
    print("k_pid_alt: [{:.5g}, {:.5g}, {:.5g}]".format(*best_formatted['k_pid_alt']))
    print("k_pid_att: [{:.5g}, {:.5g}, {:.5g}]".format(*best_formatted['k_pid_att']))
    print("k_pid_hsp: [{:.5g}, {:.5g}, {:.5g}]".format(*best_formatted['k_pid_hsp']))
    print("k_pid_vsp: [{:.5g}, {:.5g}, {:.5g}]".format(*best_formatted['k_pid_vsp']))
    print("Best target value: {:.4f}".format(optimizer.max['target']))

    with open(opt_output_path, 'w') as f:
        f.write("Best parameters found:\n")
        for k, v in best_formatted.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Best target value: {optimizer.max['target']}\n")
        f.write(f"Total optimization time: {seconds_to_hhmmss(tot_time)}\n")
        f.write(f"N iterations: {n_iter}\n")
        f.write(f"Max sim time: {simulation_time:.2f} seconds\n")

    plot_costs_trend(costs, save_path=opt_output_path.replace(".txt", "_costs.png"))

if __name__ == "__main__":
    main()
