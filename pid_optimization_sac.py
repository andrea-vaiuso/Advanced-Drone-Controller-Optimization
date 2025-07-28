"""PID Gain optimization using SAC.

This script sets up a reinforcement learning problem using a custom Gymnasium
environment in which the action space corresponds to the PID gains of the drone
controller. Each episode runs a full drone simulation with the chosen gains and
returns a reward based on the final cost. The reward is the negative of the cost
used in the Bayesian optimization script.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

import main as mainfunc
from World import World

# Load parameters and world data only once
parameters = mainfunc.load_parameters("parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()
world = World.load_world(parameters['world_data_path'])

SIMULATION_TIME = float(parameters.get('simulation_time', 150.0))


def simulate_pid(pid_gains: dict) -> tuple[float, float, float, float, float, float, bool]:
    """Run the simulation using the provided PID gains and return cost metrics."""
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
    final_time = sim.navigation_time if sim.navigation_time is not None else SIMULATION_TIME
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

    return cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, sim.has_reached_target


class PIDGainEnv(gym.Env):
    """Gym environment where actions are PID gains."""

    def __init__(self):
        super().__init__()
        # 12 continuous actions for the PID gains used by the controller
        low = np.full(12, 1e-6, dtype=np.float32)
        high = np.full(12, 1e3, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Observation is a dummy value as the environment is stateless
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        gains = {
            'k_pid_pos': (float(action[0]), float(action[1]), float(action[2])),
            'k_pid_alt': (float(action[3]), float(action[4]), float(action[5])),
            'k_pid_att': (float(action[6]), float(action[7]), float(action[8])),
            'k_pid_yaw': (0.5, 1e-6, 0.1),
            'k_pid_hsp': (float(action[9]), float(action[10]), float(action[11])),
            'k_pid_vsp': (float(action[9]), float(action[10]), float(action[11])),
        }
        cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, reached = simulate_pid(gains)
        reward = -cost
        info = {
            'final_time': final_time,
            'final_distance': final_distance,
            'pitch_osc': pitch_osc,
            'roll_osc': roll_osc,
            'thrust_osc': thrust_osc,
            'target_reached': reached,
        }
        observation = np.zeros(1, dtype=np.float32)
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info


def main():
    env = PIDGainEnv()
    model = SAC('MlpPolicy', env, verbose=1)
    # This is just a demonstration run; adjust timesteps as needed
    model.learn(total_timesteps=10)


if __name__ == '__main__':
    main()
