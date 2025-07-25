# Drone-Controller-Trajectory-Noise Optimization

## Description
This project simulates a quadcopter with PID control to analyze how trajectory noise affects control stability. It uses Multi‑Disciplinary Optimization techniques to study rotor performance and environmental noise. Scripts load waypoints, noise models, and a neural-network rotor model for realistic simulations.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/andrea-vaiuso/Advanced-Drone-Controller-Optimization.git
   cd Advanced-Drone-Controller-Optimization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main simulation:
```bash
python main.py
```

Tune PID gains with Bayesian optimization:
```bash
python pid_optimization.py
```

Edit world data using the optional GUI:
```bash
python world_creation_gui.py
```

## Project Structure
- `main.py` – entry point for the simulation.
- `Controller.py` – PID and high-level controller classes.
- `Drone.py` – physical model of the quadcopter.
- `Simulation.py` – runs the simulation loop and noise modeling.
- `Rotor/` – rotor model code and pretrained weights.
- `Noise/` – data and models for noise effects.
- `pid_optimization.py` – script for Bayesian PID tuning.
- `world_creation_gui.py` – simple GUI world editor.

## Technologies Used
- Python 3
- NumPy, SciPy, PyTorch, scikit-learn, matplotlib
- `bayesian-optimization` for tuning
- Neural network rotor model in `Rotor/TorchRotorModel.py`
