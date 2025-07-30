"""Shared utility functions for optimization scripts."""

import json
from datetime import datetime
from time import time
from typing import Dict, Any

import numpy as np
from Simulation import Simulation

# timers used for logging
start_time = time()
last_time = start_time


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


def calculate_costs(sim: Simulation, simulation_time: float) -> tuple:
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

    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0])))
    roll_osc = np.sum(np.abs(np.diff(angles[:, 1])))
    thrust_osc = np.sum(np.abs(np.diff(sim.thrust_history))) * 1e-5
    osc_weight = 3.0

    perc_completed = sim.current_seg_idx / len(sim.waypoints)

    time_cost = final_time
    final_distance_cost = final_distance ** 0.9
    oscillation_cost = osc_weight * (pitch_osc + roll_osc + thrust_osc)
    completition_cost = 1000 * (1 - perc_completed)

    cost = time_cost + final_distance_cost + oscillation_cost + completition_cost

    if not sim.has_moved:
        cost += 1000

    return (
        cost,
        time_cost,
        final_distance_cost,
        oscillation_cost,
        completition_cost,
        sim.current_seg_idx,
        len(sim.waypoints),
    )


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to a ``HH:MM:SS`` formatted string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"
