# Author: Andrea Vaiuso
# Version: 1.1
# Date: 31.07.2025
# Description: This module implements Particle Swarm Optimization (PSO) for PID gain tuning of a drone controller.

import json
import os
from datetime import datetime
from time import time

import numpy as np
import yaml

from World import World
import main as mainfunc
from Simulation import Simulation

import opt_func
from opt_func import log_step, calculate_costs, seconds_to_hhmmss, plot_costs_trend

costs = []
best_costs = []


def simulate_pid(pid_gains, parameters, waypoints, world, thrust_max, simulation_time):
    """Run a simulation with the given PID gains and return the cost metrics."""
    init_state = mainfunc.create_initial_state()
    quad_controller = mainfunc.create_quadcopter_controller(init_state=init_state,
                                                            pid_gains=pid_gains,
                                                            t_max=thrust_max,
                                                            parameters=parameters)
    
    drone = mainfunc.create_quadcopter_model(init_state=init_state,
                                             quad_controller=quad_controller,
                                             parameters=parameters)
    
    sim = mainfunc.create_simulation(drone=drone,
                                     world=world,
                                     waypoints=waypoints,
                                     parameters=parameters,
                                     noise_model=None)
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True, use_static_target=True)
    return calculate_costs(sim, simulation_time)


# initialize logging timers in opt_func
opt_func.start_time = time()
opt_func.last_time = opt_func.start_time

def decode_particle(vec: np.ndarray) -> dict:
    """Convert a particle vector into a PID gain dictionary."""
    return {
        'k_pid_pos': (vec[0], vec[1], vec[2]),
        'k_pid_alt': (vec[3], vec[4], vec[5]),
        'k_pid_att': (vec[6], vec[7], vec[8]),
        'k_pid_yaw': (0.5, 1e-6, 0.1),
        'k_pid_hsp': (vec[9], vec[10], vec[11]),
        'k_pid_vsp': (vec[12], vec[13], vec[14])
    }

def main():
    """Run PID optimization using Particle Swarm Optimization."""
    # Load settings
    with open(os.path.join('Settings', 'pso_opt.yaml'), 'r') as f:
        pso_cfg = yaml.safe_load(f)
    parameters = mainfunc.load_parameters('Settings/simulation_parameters.yaml')

    n_iter = int(pso_cfg.get('n_iter', 100))
    swarm_size = int(pso_cfg.get('swarm_size', 30))
    w = float(pso_cfg.get('inertia_weight', 0.7))
    c1 = float(pso_cfg.get('cognitive_coeff', 1.5))
    c2 = float(pso_cfg.get('social_coeff', 1.5))
    simulation_time = float(pso_cfg.get('simulation_time', 150))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('Optimizations', 'PSO', timestamp)
    os.makedirs(base_dir, exist_ok=True)
    opt_output_path = os.path.join(base_dir, 'best_parameters.txt')
    log_path = os.path.join(base_dir, 'optimization_log.json')

    thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
    waypoints = mainfunc.create_training_waypoints()
    world = World.load_world(parameters['world_data_path'])

    # Parameter bounds loaded from configuration
    pbounds_cfg = pso_cfg.get('pbounds', {})
    pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}

    lower_bounds = np.array([v[0] for v in pbounds.values()], dtype=float)
    upper_bounds = np.array([v[1] for v in pbounds.values()], dtype=float)
    dim = len(lower_bounds)

    rng = np.random.default_rng(42)
    particles_pos = rng.uniform(lower_bounds, upper_bounds, size=(swarm_size, dim))
    particles_vel = np.zeros((swarm_size, dim))

    # Use current PID gains as one of the particles
    current_best = mainfunc.load_pid_gains(parameters)
    init_particle = np.array([
        current_best['k_pid_pos'][0], current_best['k_pid_pos'][1], current_best['k_pid_pos'][2],
        current_best['k_pid_alt'][0], current_best['k_pid_alt'][1], current_best['k_pid_alt'][2],
        current_best['k_pid_att'][0], current_best['k_pid_att'][1], current_best['k_pid_att'][2],
        current_best['k_pid_hsp'][0], current_best['k_pid_hsp'][1], current_best['k_pid_hsp'][2],
        current_best['k_pid_vsp'][0], current_best['k_pid_vsp'][1], current_best['k_pid_vsp'][2]
    ])
    particles_pos[0] = np.clip(init_particle, lower_bounds, upper_bounds)

    personal_best_pos = particles_pos.copy()
    personal_best_cost = np.full(swarm_size, np.inf)
    global_best_pos = None
    global_best_cost = np.inf

    start_opt = time()
    print("Starting Particle Swarm Optimization...")

    try:
        # Main optimization loop
        for gen in range(n_iter):
            for i in range(swarm_size):
                gains = decode_particle(particles_pos[i])
                costs_sim = simulate_pid(gains, parameters, waypoints, world, thrust_max, simulation_time)
                total_cost = costs_sim['total_cost']
                costs.append(total_cost)
                log_step(gains, total_cost, log_path)
                if total_cost < personal_best_cost[i]:
                    personal_best_cost[i] = total_cost
                    personal_best_pos[i] = particles_pos[i].copy()
                if total_cost < global_best_cost:
                    global_best_cost = total_cost
                    global_best_pos = particles_pos[i].copy()
            # Update velocity and position
            for i in range(swarm_size):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                particles_vel[i] = (w * particles_vel[i] +
                                    c1 * r1 * (personal_best_pos[i] - particles_pos[i]) +
                                    c2 * r2 * (global_best_pos - particles_pos[i]))
                particles_pos[i] = particles_pos[i] + particles_vel[i]
                particles_pos[i] = np.clip(particles_pos[i], lower_bounds, upper_bounds)
            best_costs.append(global_best_cost)
            print(f"Generation {gen+1}/{n_iter} | best_cost={global_best_cost:.4f}")
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        tot_time = time() - start_opt
        best_params = decode_particle(global_best_pos)

        print("Best parameters found:")
        print("k_pid_yaw: (0.5, 1e-6, 0.1)")
        print("k_pid_pos: [{:.5g}, {:.5g}, {:.5g}]".format(*best_params['k_pid_pos']))
        print("k_pid_alt: [{:.5g}, {:.5g}, {:.5g}]".format(*best_params['k_pid_alt']))
        print("k_pid_att: [{:.5g}, {:.5g}, {:.5g}]".format(*best_params['k_pid_att']))
        print("k_pid_hsp: [{:.5g}, {:.5g}, {:.5g}]".format(*best_params['k_pid_hsp']))
        print("k_pid_vsp: [{:.5g}, {:.5g}, {:.5g}]".format(*best_params['k_pid_vsp']))
        print("Best cost value: {:.4f}".format(global_best_cost))

        with open(opt_output_path, 'w') as f:
            f.write("Best parameters found:\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"Best cost value: {global_best_cost}\n")
            f.write(f"Total optimization time: {seconds_to_hhmmss(tot_time)}\n")
            f.write(f"N iterations: {n_iter}\n")
            f.write(f"Max sim time: {simulation_time:.2f} seconds\n")

        plot_costs_trend(costs, save_path=opt_output_path.replace(".txt", "_costs.png"))
        plot_costs_trend(best_costs, save_path=opt_output_path.replace(".txt", "_best_costs.png"))

if __name__ == "__main__":
    main()
