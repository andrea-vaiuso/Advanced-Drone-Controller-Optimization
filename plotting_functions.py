# Author: Andrea Vaiuso
# Version: 2.0
# Date: 15.07.2025
# Description: This module provides functions for plotting simulation data, saving log data to CSV,
# and creating 3D animations of a drone's trajectory and attitude over time.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import euler_to_rot

def plotLogData(log_dict, time, waypoints=None, ncols=2):
    """
    Dynamically plot simulation data with per-plot legend/grid flags and configurable columns.
    
    Parameters:
        log_dict (dict):
            Keys are subplot titles (str).
            Values are dict with:
                - 'data': numpy array (1D or 2D) or dict of 1D arrays
                - 'ylabel': str for the Y-axis label
                - Optional styling for 1D/2D arrays:
                    - 'color', 'linestyle', 'label'
                    - OR 'colors', 'linestyles', 'labels'
                - Optional styling for dict-of-series:
                    - 'styles': { series_label: {'color','linestyle','label'} }
                - 'showlegend': bool (default True)
                - 'showgrid':  bool (default False)
        time (array-like): 1D array of time stamps (s).
        waypoints (list of dict, optional):
            If provided, and subplot title contains 'Position',
            draws horizontal lines at each wp['x'], wp['y'], wp['z'].
        ncols (int): number of columns for subplot grid (default 2).
    """
    # Calculate rows needed based on number of plots
    n_plots = len(log_dict)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    axes = axes.flatten()

    for ax, (title, spec) in zip(axes, log_dict.items()):
        data = spec['data']
        ylabel = spec.get('ylabel', '')
        showleg = spec.get('showlegend', True)
        showgrid = spec.get('showgrid', False)

        def _plot_series(x, y, lbl=None, color=None, ls=None):
            ax.plot(x, y, label=lbl, color=color, linestyle=ls)

        # Plot each series
        if isinstance(data, dict):
            styles = spec.get('styles', {})
            for lbl, series in data.items():
                s = styles.get(lbl, {})
                _plot_series(time, series,
                             s.get('label', lbl),
                             s.get('color'),
                             s.get('linestyle'))
        else:
            arr = np.array(data)
            if arr.ndim == 1:
                _plot_series(time, arr,
                             spec.get('label', title),
                             spec.get('color'),
                             spec.get('linestyle'))
            elif arr.ndim == 2:
                n = arr.shape[1]
                colors = spec.get('colors', [None]*n)
                linestyles = spec.get('linestyles', [None]*n)
                labels = spec.get('labels', [f"{title} {i+1}" for i in range(n)])
                for i in range(n):
                    _plot_series(time, arr[:, i],
                                 labels[i],
                                 colors[i],
                                 linestyles[i])

        # Plot waypoints for Position plots
        if waypoints and 'Position' in title:
            axis_key = title[0].lower()
            for i, wp in enumerate(waypoints):
                ax.axhline(y=wp[axis_key],
                           linestyle='--',
                           color='r',
                           label=(f"Waypoint {axis_key.upper()}" if i == 0 else None))

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        if showgrid:
            ax.grid(True)
        if showleg:
            ax.legend()

    # Remove unused axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()



def saveLogData(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, spl_history, swl_history, waypoints, filename):
    """
    Save the drone simulation data to a CSV file.
    """
    import pandas as pd

    data = {
        'Time': time_history,
        'X Position': positions[:, 0],
        'Y Position': positions[:, 1],
        'Z Position': positions[:, 2],
        'Pitch': angles_history[:, 0],
        'Roll': angles_history[:, 1],
        'Yaw': angles_history[:, 2],
        'RPM1': rpms_history[:, 0],
        'RPM2': rpms_history[:, 1],
        'RPM3': rpms_history[:, 2],
        'RPM4': rpms_history[:, 3],
        'Horizontal Speed': horiz_speed_history,
        'Vertical Speed': vertical_speed_history,
        'SPL': spl_history,
        'SWL': swl_history,
        'Waypoints': [f"{wp['x']},{wp['y']},{wp['z']}" for wp in waypoints]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def plot3DAnimation(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, 
                    targets, waypoints, start_position, dt, frame_skip,
                    window=(100, 100, 100)):
    """
    Plot a 3D animation of the drone's trajectory and attitude over time.
    """
    # --- Animation: 3D Trajectory of the Drone ---
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(0, window[0])
    ax_anim.set_ylim(0, window[1])
    ax_anim.set_zlim(0, window[2])
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_zlabel('Z')
    ax_anim.set_title('Quadcopter Animation')

    trajectory_line, = ax_anim.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax_anim.scatter([], [], [], color='red', s=50, label='Drone')
    # Add a scatter for the dynamic target
    target_scatter = ax_anim.scatter([], [], [], marker='*', color='magenta', s=100, label='Target')
    
    time_text = ax_anim.text2D(0.05, 0.05, "", transform=ax_anim.transAxes, fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
    
    # Display the starting point
    ax_anim.scatter(start_position[0], start_position[1], start_position[2],
                    marker='o', color='green', s=100, label='Start')
    # Display waypoints
    for i, wp in enumerate(waypoints, start=1):
        ax_anim.scatter(wp['x'], wp['y'], wp['z'], marker='X', color='purple', s=100,
                        label=f'Waypoint {i}' if i == 1 else None)
        ax_anim.text(wp['x'], wp['y'], wp['z'] + 2, f'{i}', color='black', fontsize=12, ha='center')

    def init_anim():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        target_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, target_scatter, time_text

    def update_anim(frame):
        nonlocal targets
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)

        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Update the dynamic target marker
        targ = targets[frame]
        target_scatter._offsets3d = ([targ[0]], [targ[1]], [targ[2]])

        # Update the arrows indicating the drone's attitude
        for q in current_quivers:
            q.remove()
        current_quivers.clear()

        phi, theta, psi = angles_history[frame]
        R = euler_to_rot(phi, theta, psi)
        arrow_len = 4
        x_body = R @ np.array([1, 0, 0])
        y_body = R @ np.array([0, 1, 0])
        z_body = R @ np.array([0, 0, 1])
        qx = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * x_body[0], arrow_len * x_body[1], arrow_len * x_body[2],
                            color='r')
        qy = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * y_body[0], arrow_len * y_body[1], arrow_len * y_body[2],
                            color='g')
        qz = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * z_body[0], arrow_len * z_body[1], arrow_len * z_body[2],
                            color='b')
        current_quivers.extend([qx, qy, qz])

        current_time = frame * dt * frame_skip
        current_rpm = rpms_history[frame]
        text_str = (f"Time: {current_time:.2f} s\n"
                    f"RPM: [{current_rpm[0]:.2f}, {current_rpm[1]:.2f}, {current_rpm[2]:.2f}, {current_rpm[3]:.2f}]\n"
                    f"Vertical Speed: {vertical_speed_history[frame]:.2f} m/s\n"
                    f"Horizontal Speed: {horiz_speed_history[frame]:.2f} m/s\n"
                    f"Pitch: {angles_history[frame][0]:.4f} rad\n"
                    f"Roll: {angles_history[frame][1]:.4f} rad\n"
                    f"Yaw: {angles_history[frame][2]:.4f} rad")
        time_text.set_text(text_str)

        return trajectory_line, drone_scatter, target_scatter, time_text, *current_quivers

    # List to manage attitude arrow objects
    current_quivers = []

    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(positions),
                                  init_func=init_anim, interval=50, blit=False, repeat=True)
    plt.show()

