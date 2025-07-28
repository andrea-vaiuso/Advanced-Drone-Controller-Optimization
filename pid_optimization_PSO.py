import json
import os
from datetime import datetime
from time import time

import numpy as np

from World import World
import main as mainfunc

# PSO settings
n_iter = 200  # number of iterations
num_particles = 20
w = 0.5
c1 = 1.5
c2 = 1.5

iteration = 0
best_cost = float('inf')
best_params = None
simulation_time = 150.0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join('Optimizations', 'PSO', timestamp)
os.makedirs(base_dir, exist_ok=True)
opt_output_path = os.path.join(base_dir, 'best_parameters.txt')
log_path = os.path.join(base_dir, 'optimization_log.json')

start_time = time()
last_time = start_time

parameters = mainfunc.load_parameters("parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()

world = World.load_world(parameters['world_data_path'])


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


def simulate_pid(pid_gains: dict) -> float:
    """Run a simulation with the given PID gains and return the cost."""
    init_state = mainfunc.create_initial_state()
    quad_controller = mainfunc.create_quadcopter_controller(
        init_state=init_state,
        pid_gains=pid_gains,
        t_max=thrust_max,
        parameters=parameters,
    )
    drone = mainfunc.create_quadcopter_model(
        init_state=init_state,
        quad_controller=quad_controller,
        parameters=parameters,
    )
    sim = mainfunc.create_simulation(
        drone=drone,
        world=world,
        waypoints=waypoints,
        parameters=parameters,
        noise_model=None,
    )
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True)

    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time
    final_target = waypoints[-1]
    final_distance = np.linalg.norm(
        sim.drone.state['pos'] - np.array([final_target['x'], final_target['y'], final_target['z']])
    )
    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0])))
    roll_osc = np.sum(np.abs(np.diff(angles[:, 1])))
    thrust_osc = np.sum(np.abs(np.diff(sim.thrust_history))) * 1e-5
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc + thrust_osc)
    if not sim.has_moved:
        cost += 1000
    elif not sim.has_reached_target:
        cost += 1000

    return cost


def pso_optimize(pbounds: dict) -> None:
    """Perform the Particle Swarm Optimization search."""
    global iteration, best_cost, best_params
    names = list(pbounds.keys())
    low = np.array([pbounds[n][0] for n in names])
    high = np.array([pbounds[n][1] for n in names])
    dim = len(names)

    positions = np.random.uniform(low, high, (num_particles, dim))
    velocities = np.zeros_like(positions)

    def vector_to_dict(vec):
        vals = {name: float(value) for name, value in zip(names, vec)}
        return {
            'k_pid_pos': (vals['kp_pos'], vals['ki_pos'], vals['kd_pos']),
            'k_pid_alt': (vals['kp_alt'], vals['ki_alt'], vals['kd_alt']),
            'k_pid_att': (vals['kp_att'], vals['ki_att'], vals['kd_att']),
            'k_pid_yaw': (0.5, 1e-6, 0.1),
            'k_pid_hsp': (vals['kp_hsp'], vals['ki_hsp'], vals['kd_hsp']),
            'k_pid_vsp': (vals['kp_vsp'], vals['ki_vsp'], vals['kd_vsp']),
        }

    personal_best = positions.copy()
    personal_best_cost = np.full(num_particles, np.inf)
    global_best = None
    global_best_cost = np.inf

    # Initial evaluation
    for i in range(num_particles):
        params = vector_to_dict(positions[i])
        c = simulate_pid(params)
        log_step(params, c)
        personal_best_cost[i] = c
        if c < global_best_cost:
            global_best_cost = c
            global_best = positions[i].copy()
        iteration += 1

    for it in range(n_iter):
        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - positions[i])
                + c2 * r2 * (global_best - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], low, high)

            params = vector_to_dict(positions[i])
            c = simulate_pid(params)
            log_step(params, c)
            iteration += 1

            if c < personal_best_cost[i]:
                personal_best_cost[i] = c
                personal_best[i] = positions[i].copy()
                if c < global_best_cost:
                    global_best_cost = c
                    global_best = positions[i].copy()

        print(f"Iter {it+1}/{n_iter}: best cost={global_best_cost:.4f}")

    best_params = vector_to_dict(global_best)
    best_cost = global_best_cost


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to a ``HH:MM:SS`` formatted string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def main() -> None:
    """Entry point for running the PSO PID gain optimization."""
    pbounds = {
        'kp_pos': (1e-6, 1e3), 'ki_pos': (1e-6, 1e3), 'kd_pos': (1e-6, 1e3),
        'kp_alt': (1e-6, 1e3), 'ki_alt': (1e-6, 1e3), 'kd_alt': (1e-6, 1e3),
        'kp_att': (1e-6, 1e3), 'ki_att': (1e-6, 1e3), 'kd_att': (1e-6, 1e3),
        'kp_hsp': (1e-6, 1e3), 'ki_hsp': (1e-6, 1e3), 'kd_hsp': (1e-6, 1e3),
        'kp_vsp': (1e-6, 1e3), 'ki_vsp': (1e-6, 1e3), 'kd_vsp': (1e-6, 1e3),
    }

    total_start = time()
    print("Starting PSO optimization...")
    pso_optimize(pbounds)
    tot_time = time() - total_start
    print(f"Optimization completed in {tot_time:.2f} seconds.")

    with open(opt_output_path, 'w') as f:
        f.write("Best parameters found:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Best cost: {best_cost}\n")
        f.write(f"Total optimization time: {seconds_to_hhmmss(tot_time)}\n")
        f.write(f"N iterations: {n_iter}\n")
        f.write(f"Max sim time: {simulation_time:.2f} seconds\n")


if __name__ == '__main__':
    main()
