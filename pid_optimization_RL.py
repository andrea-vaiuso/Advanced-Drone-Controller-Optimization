# Author: Andrea Vaiuso
# Version: 1.0
# Date: 31.07.2025
# Description: This module implements a TD3-based Reinforcement Learning approach
#              for PID gain tuning of a drone controller. The implementation is
#              inspired by the methodology presented in "Processes 2025, 13, 735"
#              and follows the structure of the existing optimization scripts.

"""TD3 optimization for PID tuning.

This script trains a TD3 agent to propose new sets of PID gains for the
drone controller. Each action of the agent corresponds to a complete PID
configuration (with yaw PIDs fixed), while the environment returns the
negative of the simulation cost as reward. The goal is to minimize the cost
metrics defined in :mod:`opt_func`.

Logs, results and plots are stored using the same format as in the other
optimization scripts.
"""

from __future__ import annotations

import os
from datetime import datetime
from time import time
from typing import Dict

import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from World import World
import main as mainfunc
from Simulation import Simulation

import opt_func
from opt_func import (
    log_step,
    calculate_costs,
    seconds_to_hhmmss,
    plot_costs_trend,
    show_best_params
)


# ---------------------------------------------------------------------------
# Configuration and global objects
# ---------------------------------------------------------------------------

with open(os.path.join("Settings", "td3_opt.yaml"), "r") as f:
    td3_cfg = yaml.safe_load(f)

simulation_time = float(td3_cfg.get("simulation_time", 150))
total_timesteps = int(td3_cfg.get("total_timesteps", 1000))
action_noise_sigma = float(td3_cfg.get("action_noise_sigma", 0.1))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join("Optimizations", "TD3", timestamp)
os.makedirs(base_dir, exist_ok=True)

opt_output_path = os.path.join(base_dir, "best_parameters.txt")
log_path = os.path.join(base_dir, "optimization_log.json")

# initialize logging timers in opt_func
opt_func.start_time = time()
opt_func.last_time = opt_func.start_time

parameters = mainfunc.load_parameters("Settings/simulation_parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()
world = World.load_world(parameters["world_data_path"])
noise_model = mainfunc.load_dnn_noise_model(parameters)

pbounds_cfg = td3_cfg.get("pbounds", {})
PB_NAMES = list(pbounds_cfg.keys())
PB_LOW = np.array([pbounds_cfg[k][0] for k in PB_NAMES], dtype=np.float32)
PB_HIGH = np.array([pbounds_cfg[k][1] for k in PB_NAMES], dtype=np.float32)
DIM = len(PB_NAMES)

costs: list[float] = []
best_costs: list[float] = []
best_cost = np.inf
best_params: Dict[str, tuple] | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def simulate_pid(pid_gains: Dict[str, tuple]) -> Dict[str, float]:
    """Run a simulation with the given PID gains and return the cost metrics."""
    init_state = mainfunc.create_initial_state()
    quad_controller = mainfunc.create_quadcopter_controller(
        init_state=init_state,
        pid_gains=pid_gains,
        t_max=thrust_max,
        parameters=parameters,
    )
    drone = mainfunc.create_quadcopter_model(
        init_state=init_state, quad_controller=quad_controller, parameters=parameters
    )
    sim = mainfunc.create_simulation(
        drone=drone,
        world=world,
        waypoints=waypoints,
        parameters=parameters,
        noise_model=noise_model,
        generate_sound_map=False,
    )
    sim.startSimulation(
        stop_at_target=True, verbose=False, stop_sim_if_not_moving=True, use_static_target=True
    )
    return calculate_costs(sim, simulation_time)


def decode_action(vec: np.ndarray) -> Dict[str, tuple]:
    """Convert an action vector into a dictionary of PID gains."""
    return {
        "k_pid_pos": (vec[0], vec[1], vec[2]),
        "k_pid_alt": (vec[3], vec[4], vec[5]),
        "k_pid_att": (vec[6], vec[7], vec[8]),
        "k_pid_yaw": (0.5, 1e-6, 0.1),
        "k_pid_hsp": (vec[9], vec[10], vec[11]),
        "k_pid_vsp": (vec[12], vec[13], vec[14]),
    }


# ---------------------------------------------------------------------------
# Environment definition
# ---------------------------------------------------------------------------


class PIDEnv(gym.Env):
    """Gym environment where each action is a set of PID gains.

    The observation is the last action taken. Episodes last a single step
    (bandit-like setup). The reward is the negative total cost returned by the
    simulator. Logging and best-parameter tracking are handled internally.
    """

    metadata = {"render.modes": []}

    def __init__(self, verbose: bool = True, set_initial_obs: bool = True):
        super().__init__()
        self.action_space = spaces.Box(PB_LOW, PB_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(PB_LOW, PB_HIGH, dtype=np.float32)
        current_best = mainfunc.load_pid_gains(parameters)
        self.initial_obs = np.array(
            [
                current_best["k_pid_pos"][0],
                current_best["k_pid_pos"][1],
                current_best["k_pid_pos"][2],
                current_best["k_pid_alt"][0],
                current_best["k_pid_alt"][1],
                current_best["k_pid_alt"][2],
                current_best["k_pid_att"][0],
                current_best["k_pid_att"][1],
                current_best["k_pid_att"][2],
                current_best["k_pid_hsp"][0],
                current_best["k_pid_hsp"][1],
                current_best["k_pid_hsp"][2],
                current_best["k_pid_vsp"][0],
                current_best["k_pid_vsp"][1],
                current_best["k_pid_vsp"][2],
            ],
            dtype=np.float32,
        )
        self.set_initial_obs = set_initial_obs
        if self.set_initial_obs:
            self.state = self.initial_obs.copy()
        else:
            self.state = np.zeros(DIM, dtype=np.float32)
        self.current_step = 0
        self.verbose = verbose

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if self.set_initial_obs:
            self.state = self.initial_obs.copy()
        else:
            self.state = np.zeros(DIM, dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        """Take a step in the environment with the given action."""
        self.current_step += 1
        action = np.clip(action, PB_LOW, PB_HIGH)
        gains = decode_action(action)
        sim_costs = simulate_pid(gains)
        total_cost = sim_costs["total_cost"]
        reward = -total_cost

        # Logging and best tracking
        log_step(gains, total_cost, log_path, sim_costs)
        costs.append(total_cost)
        global best_cost, best_params
        if total_cost < best_cost:
            best_cost = total_cost
            best_params = gains
        best_costs.append(best_cost)

        self.state = action.astype(np.float32)
        terminated = True  # one-step episode
        truncated = False
        info = {"costs": sim_costs}
        if self.verbose: print(f"[ TD3 ] Step ({self.current_step}/{total_timesteps}) completed: total_cost={total_cost:.4f}, best_cost={best_cost:.4f}, costs: {sim_costs}")
        return self.state, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Main optimization routine
# ---------------------------------------------------------------------------


def main() -> None:
    """Run PID optimization using TD3."""

    env = PIDEnv(verbose=True)

    # Action noise for exploration
    action_noise = NormalActionNoise(
        mean=np.zeros(DIM), sigma=action_noise_sigma * np.ones(DIM)
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
    )

    start_opt = time()
    print("Starting TD3 optimization...")
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        tot_time = time() - start_opt

        if best_params is None:
            print("No evaluations were performed.")
            return

        show_best_params(best_params, opt_output_path, best_cost, len(costs), simulation_time, tot_time)

        # Save plots of cost trends
        plot_costs_trend(costs, save_path=opt_output_path.replace(".txt", "_costs.png"))
        plot_costs_trend(best_costs, save_path=opt_output_path.replace(".txt", "_best_costs.png"))


if __name__ == "__main__":
    main()

