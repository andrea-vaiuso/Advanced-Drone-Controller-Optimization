import numpy as np
from bayes_opt import BayesianOptimization
from World import World
import main as mainfunc

# Optimization parameters
iteration = 0
n_iter = 200
costs = []
best_target = -float('inf')
best_params = None
best_file = 'best_pid.txt'
simulation_time = 150.0 

parameters = mainfunc.load_parameters("parameters.yaml")
thrust_max = mainfunc.get_max_thrust_from_rotor_model(parameters)
waypoints = mainfunc.create_training_waypoints()

world = World.load_world(parameters['world_data_path'])


def simulate_pid(pid_gains):

    # Initial drone state
    init_state = mainfunc.create_initial_state()

    quad_controller = mainfunc.create_quadcopter_controller(init_state=init_state, pid_gains=pid_gains, t_max=thrust_max, parameters=parameters)
    # Initialize the quadcopter model with the rotor model
    # Default values are taken from the paper: "Modeling of a Quadcopter Trajectory Tracking System Using PID Controller" by Sabir et.al. (2020)
    drone = mainfunc.create_quadcopter_model(init_state=init_state, quad_controller=quad_controller, parameters=parameters)

    # Initialize the simulation
    sim = mainfunc.create_simulation(drone=drone, world=world, waypoints=waypoints, parameters=parameters, noise_model=None)

    # sim.setWind(max_simulation_time=simulation_time, dt=dt, height=100, airspeed=10, turbulence_level=10, plot_wind_signal=False, seed = None)
    sim.startSimulation(stop_at_target=True, verbose=False, stop_sim_if_not_moving=True)

    # Collect results
    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time
    
    final_target = {'x': waypoints[-1]['x'], 'y': waypoints[-1]['y'], 'z': waypoints[-1]['z']}
    final_distance = np.linalg.norm(sim.drone.state['pos'] - np.array([final_target['x'],
                                                              final_target['y'],
                                                              final_target['z']]))

    pitch_osc = np.sum(np.abs(np.diff(angles[:, 0])))
    roll_osc  = np.sum(np.abs(np.diff(angles[:, 1])))
    thrust_osc = np.sum(np.abs(np.diff(sim.thrust_history))) * 1e-5
    osc_weight = 3.0

    cost = final_time + (final_distance ** 0.9) + osc_weight * (pitch_osc + roll_osc + thrust_osc)
    if not sim.has_moved: cost += 1000
    elif not sim.has_reached_target: cost += 1000

    return cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, sim.has_reached_target

def objective(kp_pos, ki_pos, kd_pos,
              kp_alt, ki_alt, kd_alt,
              kp_att, ki_att, kd_att,
              kp_hsp, ki_hsp, kd_hsp,
              kp_vsp, ki_vsp, kd_vsp):
    global iteration, best_target, best_params

    iteration += 1
    cost, final_time, final_distance, pitch_osc, roll_osc, thrust_osc, targ_reached = simulate_pid(
        pid_gains={
        'kp_pos': kp_pos, 'ki_pos': ki_pos, 'kd_pos': kd_pos,
        'kp_alt': kp_alt, 'ki_alt': ki_alt, 'kd_alt': kd_alt,
        'kp_att': kp_att, 'ki_att': ki_att, 'kd_att': kd_att,
        'kp_yaw': 0.5, 'ki_yaw': 1e-6, 'kd_yaw': 0.1,  # Fixed yaw PID gains
        'kp_hsp': kp_hsp, 'ki_hsp': ki_hsp, 'kd_hsp': kd_hsp,
        'kp_vsp': kp_vsp, 'ki_vsp': ki_vsp, 'kd_vsp': kd_vsp
    }
    )
    target = -cost  # bayes_opt massimizza

    # se questo target è migliore del best-so-far, aggiorna file
    if target > best_target:
        best_target = target
        best_params = {
            'kp_pos': kp_pos, 'ki_pos': ki_pos, 'kd_pos': kd_pos,
            'kp_alt': kp_alt, 'ki_alt': ki_alt, 'kd_alt': kd_alt,
            'kp_att': kp_att, 'ki_att': ki_att, 'kd_att': kd_att,
            'kp_yaw': 0.5, 'ki_yaw': 1e-6, 'kd_yaw': 0.1,  # Fixed yaw PID gains
            'kp_hsp': kp_hsp, 'ki_hsp': ki_hsp, 'kd_hsp': kd_hsp,
            'kp_vsp': kp_vsp, 'ki_vsp': ki_vsp, 'kd_vsp': kd_vsp,
            'final_time': final_time,
            'cost': cost
        }
        # scrivi su file
        with open(best_file, 'w') as f:
            f.write(f"Iteration: {iteration}\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"target = {best_target}\n")

    print(f"{iteration}/{n_iter}: cost={cost:.4f}, best target={best_target:.4f}, final_time={final_time:.2f}s, final_distance={final_distance:.2f}m, pitch_osc={pitch_osc:.2f}, roll_osc={roll_osc:.2f}, thrust_osc={thrust_osc:.2f}, target_reached={targ_reached}")
    return target


def main():
    # Define the bounds for the optimization variables
    pbounds = {
        'kp_pos': (1e-7, 1), 
        'ki_pos': (1e-5, 1), 
        'kd_pos': (1, 10),

        'kp_alt': (10, 200),   
        'ki_alt': (1e-1, 30),  
        'kd_alt': (10, 200),

        'kp_att': (1, 5),     
        'ki_att': (1e-5, 1),  
        'kd_att': (1, 5),

        'kp_hsp': (30, 300),
        'ki_hsp': (1e-5, 1),
        'kd_hsp': (1e-5, 1),

        'kp_vsp': (30, 300),
        'ki_vsp': (1e-5, 1),
        'kd_vsp': (1e-5, 1)
    }
    init_guess = {
        'kp_pos': 0.7,
        'ki_pos': 0.1,
        'kd_pos': 5.25,
        'kp_alt': 150.1,
        'ki_alt': 20,
        'kd_alt': 100,
        'kp_att': 1.7,
        'ki_att': 1e-4,
        'kd_att': 1.25,
        'kp_hsp': 300,
        'ki_hsp': 0.0001,
        'kd_hsp': 0.0001,
        'kp_vsp': 100.5,
        'ki_vsp': 0.0001,
        'kd_vsp': 0.01
    }
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
        init_points=5,
        n_iter=n_iter,
    )
    print("Best parameters found:")
    best = optimizer.max['params']

    print("kp_yaw, ki_yaw, kd_yaw = 0.5, 1e-6, 0.1")
    print("kp_pos, ki_pos, kd_pos = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_pos'], best['ki_pos'], best['kd_pos']))
    print("kp_alt, ki_alt, kd_alt = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_alt'], best['ki_alt'], best['kd_alt']))
    print("kp_att, ki_att, kd_att = {:.5g}, {:.5g}, {:.5g}".format(
        best['kp_att'], best['ki_att'], best['kd_att']))

if __name__ == "__main__":
    main()