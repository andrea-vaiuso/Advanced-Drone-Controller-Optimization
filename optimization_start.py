"""Launch multiple PID optimization algorithms concurrently.

This script starts all available optimization algorithms using the
same configuration parameters. Each optimizer runs in its own
process so that they execute concurrently.

If the operating system supports separate terminal windows (e.g. on
Windows using the ``CREATE_NEW_CONSOLE`` flag), the optimizers can be
launched in individual command prompts. Otherwise they run as
background processes within the same terminal.
"""

from multiprocessing import Process
from typing import Type, List

import main as mainfunc
from pid_optimization_ga import GAPIDOptimizer
from pid_optimization_gwo import GWOPIDOptimizer
from pid_optimization_pso import PSOPIDOptimizer
from pid_optimization_sac import SACPIDOptimizer
from pid_optimization_td3 import TD3PIDOptimizer
from pid_optimization_bayopt import BayesianPIDOptimizer
from optimizer import Optimizer

# List of optimizer classes to run
OPTIMIZER_CLASSES: List[Type[Optimizer]] = [
    GAPIDOptimizer,
    PSOPIDOptimizer,
    GWOPIDOptimizer,
    SACPIDOptimizer,
    TD3PIDOptimizer,
    BayesianPIDOptimizer,
]


def run_optimizer(opt_class: Type[Optimizer], waypoints: list, study_name: str) -> None:
    """Instantiate and execute the optimization algorithm."""
    optimizer = opt_class(
        verbose=True,
        set_initial_obs=False,
        simulate_wind_flag=False,
        waypoints=waypoints,
        study_name=study_name
    )
    optimizer.optimize()


def main() -> None:
    """Start all optimizers in separate processes."""

    study_name = "std_wayp_no_initobs_no_wind"

    waypoints = mainfunc.create_training_waypoints()

    processes = [
        Process(target=run_optimizer, args=(opt_class, waypoints, study_name))
        for opt_class in OPTIMIZER_CLASSES
    ]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()