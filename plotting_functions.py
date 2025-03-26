from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import euler_to_rot

def plotLogData(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, waypoints):
     # --- Post-Simulation Plots (Positions, Attitude, RPM, Speeds) ---
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    
    # X Position
    axs[0, 0].plot(time_history, positions[:, 0], label='X Position')
    for wp in waypoints:
        axs[0, 0].axhline(y=wp['x'], linestyle='--', color='r', label='Waypoint X' if wp == waypoints[0] else None)
    axs[0, 0].set_title('X Position')
    axs[0, 0].set_ylabel('X (m)')
    axs[0, 0].legend()
    
    # Y Position
    axs[1, 0].plot(time_history, positions[:, 1], label='Y Position')
    for wp in waypoints:
        axs[1, 0].axhline(y=wp['y'], linestyle='--', color='r', label='Waypoint Y' if wp == waypoints[0] else None)
    axs[1, 0].set_title('Y Position')
    axs[1, 0].set_ylabel('Y (m)')
    axs[1, 0].legend()
    
    # Z Position
    axs[2, 0].plot(time_history, positions[:, 2], label='Z Position')
    for wp in waypoints:
        axs[2, 0].axhline(y=wp['z'], linestyle='--', color='r', label='Waypoint Z' if wp == waypoints[0] else None)
    axs[2, 0].set_title('Z Position')
    axs[2, 0].set_ylabel('Z (m)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].legend()
    
    # Attitude: Pitch, Roll, Yaw
    axs[0, 1].plot(time_history, angles_history[:, 0], label='Pitch')
    axs[0, 1].plot(time_history, angles_history[:, 1], label='Roll')
    axs[0, 1].plot(time_history, angles_history[:, 2], label='Yaw')
    axs[0, 1].set_title('Attitude (Pitch, Roll, Yaw)')
    axs[0, 1].set_ylabel('Angle (rad)')
    axs[0, 1].legend()
    
    # Motor RPMs
    axs[1, 1].plot(time_history, rpms_history[:, 0], label='RPM1')
    axs[1, 1].plot(time_history, rpms_history[:, 1], label='RPM2')
    axs[1, 1].plot(time_history, rpms_history[:, 2], label='RPM3')
    axs[1, 1].plot(time_history, rpms_history[:, 3], label='RPM4')
    axs[1, 1].set_title('Motor RPMs')
    axs[1, 1].set_ylabel('RPM')
    axs[1, 1].legend()
    
    # Speeds: Horizontal and Vertical
    axs[2, 1].plot(time_history, horiz_speed_history, label='Horizontal Speed', color='r')
    axs[2, 1].plot(time_history, vertical_speed_history, label='Vertical Speed', color='g')
    axs[2, 1].set_title('Speeds')
    axs[2, 1].set_ylabel('Speed (m/s)')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].legend()
    
    axs[0, 1].set_xlabel('Time (s)')
    axs[1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Drone Simulation Data vs Time', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot3DAnimation(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, targets, waypoints, start_position, dt, frame_skip):
    """
    Plot a 3D animation of the drone's trajectory and attitude over time.
    """
    
    # --- Animation: 3D Trajectory of the Drone ---
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(0, 100)
    ax_anim.set_ylim(0, 100)
    ax_anim.set_zlim(0, 100)
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