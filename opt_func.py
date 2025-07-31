"""Shared utility functions for optimization scripts."""

import json
from datetime import datetime
from time import time
from typing import Dict, Any
import matplotlib.pyplot as plt

import numpy as np
from Simulation import Simulation

# timers used for logging
start_time = time()
last_time = start_time


def plot_costs_trend(costs: list, save_path: str = None) -> None:
    """
    Plot the trend of costs over iterations. If `save_path` is provided, save the plot to that path.
    A red marker indicates the best cost found, while a blue line shows the trend.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(costs, color='blue', label='Cost Trend')
    best_cost = min(costs)
    best_idx = costs.index(best_cost)
    plt.scatter(best_idx, best_cost, color='red', label='Best Cost', zorder=5)
    plt.title('Cost Trend Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)

    plt.show()
    




def log_step(params: Dict[str, Any], cost: float, log_path: str) -> None:
    """Append a single optimization step to a JSON log file."""
    global last_time
    current = time()
    entry = {
        'target': -cost,
        'params': {
            k: ([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v))
            for k, v in params.items()
        },
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

def get_overshoot_cost(sim: Simulation, overshoot_weight = 500.0) -> float:
    """Calculate the overshoot cost based on the simulation data."""
    times = np.array(sim.time_history)
    dists = np.array(sim.distance_history)
    seg_idxs = np.array(sim.seg_idx_history)

    overshoot_integral = 0.0
    for seg in range(len(sim.waypoints)):
        mask = seg_idxs == seg
        if not np.any(mask):
            continue
        t_seg = times[mask]
        overs = np.maximum(0, dists[mask] - sim.target_shift_threshold_distance)
        integral_seg = np.trapz(overs, t_seg)
        overshoot_integral += integral_seg

    return overshoot_integral * overshoot_weight

def get_completion_cost(sim: Simulation, completion_weight = 1000.0) -> float:
    """Calculate the completion cost based on the simulation data."""
    perc_completed = sim.current_seg_idx / len(sim.waypoints)
    return completion_weight * (1 - perc_completed)


def calculate_costs(sim: Simulation, simulation_time: float, 
                    osc_weight: float = 1.0,
                    overshoot_weight: float = 0.02, 
                    completition_weight: float = 1000.0,
                    pitch_roll_oscillation_weight: float = 1.0,
                    thrust_oscillation_weight: float = 1e-5,
                    power_weight: float = 1e-4,
                    noise_weight: float = 6e-23) -> tuple:
    
    """Compute the cost metrics used for PID gain optimization."""
    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time

    final_target = {
        'x': sim.waypoints[-1]['x'],
        'y': sim.waypoints[-1]['y'],
        'z': sim.waypoints[-1]['z'],
    }
    final_distance = np.linalg.norm(
        sim.drone.state['pos'] - np.array([final_target['x'], final_target['y'], final_target['z']])
    )

    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0]))) * pitch_roll_oscillation_weight
    roll_osc = np.sum(np.abs(np.diff(angles[:, 1]))) * pitch_roll_oscillation_weight
    thrust_osc = np.sum(np.abs(np.diff(sim.thrust_history))) * thrust_oscillation_weight

    power_cost = np.sum(np.array(sim.power_history)) * power_weight


    time_cost = final_time

    final_distance_cost = final_distance ** 0.9

    oscillation_cost = osc_weight * (pitch_osc + roll_osc + thrust_osc)

    completition_cost = get_completion_cost(sim, completition_weight)

    overshoot_cost = get_overshoot_cost(sim, overshoot_weight)
    p = 12  # norm order for noise cost
    if len(sim.swl_history) > 0:
        swl = np.array(sim.swl_history, dtype=float)
        noise_cost = noise_weight * np.linalg.norm(swl, ord=p)**p
    else:
        noise_cost = 0.0

    total_cost = time_cost + \
        final_distance_cost + \
        oscillation_cost + \
        completition_cost + \
        overshoot_cost + \
        power_cost + \
        noise_cost

    if not sim.has_moved:
        total_cost += 1000

    return {
        'total_cost': total_cost,
        'time_cost': time_cost,
        'final_distance_cost': final_distance_cost,
        'oscillation_cost': oscillation_cost,
        'completition_cost': completition_cost,
        'overshoot_cost': overshoot_cost,
        'power_cost': power_cost,
        'noise_cost': noise_cost,
        'n_waypoints_completed': sim.current_seg_idx,
        'tot_waypoints': len(sim.waypoints),
    }


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to a ``HH:MM:SS`` formatted string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"
