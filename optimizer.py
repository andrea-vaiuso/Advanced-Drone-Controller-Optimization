# Author: Andrea Vaiuso
# Version: 1.0
# Date: 06.08.2025
# Description: Base class for PID gain optimizers.
"""Base optimizer class providing shared utilities for PID tuning algorithms."""

import os
from datetime import datetime
from time import time
from typing import Optional

import yaml

from World import World
import main as mainfunc
import opt_func


class Optimizer:
    """Common utilities and interface for PID optimization algorithms.

    Parameters
    ----------
    algorithm_name : str
        Name of the optimization method. Used for output folder naming.
    config_file : str
        Path to the configuration YAML file for the optimizer.
    parameters_file : str
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Use current PID gains as starting observation when ``True``.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    study_name : str, optional
        Optional suffix appended to the timestamped results folder.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is generated.
    """

    def __init__(
        self,
        algorithm_name: str,
        config_file: str,
        parameters_file: str,
        *,
        verbose: bool = True,
        set_initial_obs: bool = True,
        simulate_wind_flag: bool = False,
        study_name: str = "",
        waypoints: Optional[list] = None,
    ) -> None:
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.verbose = verbose
        self.set_initial_obs = set_initial_obs
        self.simulate_wind_flag = simulate_wind_flag

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.timestamp if not study_name else f"{self.timestamp}_{study_name}"
        self.base_dir = os.path.join("Optimizations", algorithm_name, folder)
        os.makedirs(self.base_dir, exist_ok=True)
        self.opt_output_path = os.path.join(self.base_dir, "best_parameters.txt")
        self.log_path = os.path.join(self.base_dir, "optimization_log.json")

        opt_func.start_time = time()
        opt_func.last_time = opt_func.start_time

        self.parameters = mainfunc.load_parameters(parameters_file)
        self.thrust_max = mainfunc.get_max_thrust_from_rotor_model(self.parameters)
        self.waypoints = (
            waypoints if waypoints is not None else mainfunc.create_training_waypoints()
        )
        self.world = World.load_world(self.parameters["world_data_path"])
        self.noise_model = mainfunc.load_dnn_noise_model(self.parameters)

    def optimize(self) -> None:
        """Run the optimization routine. Must be implemented by subclasses."""
        raise NotImplementedError

