# Advanced Drone Controller Optimization

<p align="center">
  <img src="Pic/anim.gif" alt="Simulation Animation" width="49%">
  <img src="Pic/logs.png" alt="Simulation Logs" width="49%">
</p>

> **Author:** Andrea Vaiuso  
> **License:** See repository  
> **Python:** 3.10+

---

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Features](#2-features)  
3. [Installation](#3-installation)  
4. [Quick Start](#4-quick-start)  
5. [Quadcopter Dynamic Model](#5-quadcopter-dynamic-model)  
   5.1 [Reference Frames and Notation](#51-reference-frames-and-notation)  
   5.2 [Translational Dynamics](#52-translational-dynamics)  
   5.3 [Rotational Dynamics](#53-rotational-dynamics)  
   5.4 [Motor Mixing](#54-motor-mixing)  
   5.5 [Neural-Network Rotor Model (BEMT)](#55-neural-network-rotor-model-bemt)  
   5.6 [Numerical Integration](#56-numerical-integration)  
6. [Cascade PID Control Architecture](#6-cascade-pid-control-architecture)  
   6.1 [PID Law](#61-pid-law)  
   6.2 [Cascade Loops](#62-cascade-loops)  
   6.3 [Anti-Windup and Saturation](#63-anti-windup-and-saturation)  
7. [Dryden Wind Turbulence Model](#7-dryden-wind-turbulence-model)  
8. [Acoustic Noise Model](#8-acoustic-noise-model)  
   8.1 [Lookup-Table Sound Model (DNN)](#81-lookup-table-sound-model-dnn)  
   8.2 [EMPA Regression Model](#82-empa-regression-model)  
   8.3 [Propagation and Atmospheric Absorption](#83-propagation-and-atmospheric-absorption)  
9. [Psychoacoustic Annoyance](#9-psychoacoustic-annoyance)  
10. [World and Environment Model](#10-world-and-environment-model)  
11. [Cost Function for Optimization](#11-cost-function-for-optimization)  
12. [Optimization Algorithms](#12-optimization-algorithms)  
    12.1 [Bayesian Optimization](#121-bayesian-optimization)  
    12.2 [Particle Swarm Optimization (PSO)](#122-particle-swarm-optimization-pso)  
    12.3 [Genetic Algorithm (GA)](#123-genetic-algorithm-ga)  
    12.4 [Grey Wolf Optimizer (GWO)](#124-grey-wolf-optimizer-gwo)  
    12.5 [Soft Actor–Critic (SAC)](#125-soft-actorcritic-sac)  
    12.6 [Twin-Delayed DDPG (TD3)](#126-twin-delayed-ddpg-td3)  
13. [Configuration Reference](#13-configuration-reference)  
14. [Project Structure](#14-project-structure)  
15. [Usage Guide](#15-usage-guide)  
    15.1 [Running a Simulation](#151-running-a-simulation)  
    15.2 [Running a Single Optimizer](#152-running-a-single-optimizer)  
    15.3 [Running All Optimizers Concurrently](#153-running-all-optimizers-concurrently)  
    15.4 [World Creation GUI](#154-world-creation-gui)  
16. [Output and Results](#16-output-and-results)  
17. [References](#17-references)

---

## 1. Introduction

This project provides a high-fidelity simulation environment for a quadcopter UAV and a suite of metaheuristic and reinforcement-learning algorithms for automatically tuning the gains of its cascade PID controller. The simulation couples rigid-body flight dynamics with a neural-network rotor model trained on Blade Element Momentum Theory (BEMT) data, a data-driven acoustic emission model with psychoacoustic annoyance evaluation, and an optional Dryden atmospheric turbulence generator. The tuning objective is formulated as a multi-component scalar cost that balances navigation performance (time, tracking accuracy, overshoot), control quality (attitude and thrust oscillations), energy consumption, and community noise impact.

---

## 2. Features

| Category | Capability |
|---|---|
| **Dynamics** | 6-DOF rigid-body model, RK4 integration, ground-contact logic |
| **Rotor model** | PyTorch DNN trained on BEMT data — predicts $(T,\,Q,\,P,\,C_T,\,C_Q,\,C_P)$ from RPM |
| **Control** | Three-stage cascade PID: Position → Velocity → Attitude, with yaw decoupling |
| **Wind** | Dryden spectral turbulence model (longitudinal, lateral, vertical) |
| **Noise** | Angle-/RPM-dependent SWL lookup table, ISO 9613-1 atmospheric absorption, spherical spreading |
| **Psychoacoustics** | Zwicker loudness (ISO 532-1), DIN 45692 sharpness, Daniel & Weber roughness, Zwicker PA |
| **Optimizers** | Bayesian Optimization, PSO, GA, GWO, SAC (RL), TD3 (RL) |
| **Environment** | Grid-based world with semantic area types (housing, industrial, open field, forbidden) |
| **Visualization** | 3D trajectory animation, time-series log plots, spatial noise emission maps and histograms |

---

## 3. Installation

```bash
git clone https://github.com/andrea-vaiuso/Advanced-Drone-Controller-Optimization.git
cd Advanced-Drone-Controller-Optimization
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and animation |
| `pandas` | Data manipulation |
| `pillow` | Image I/O for world backgrounds |
| `torch` | Rotor model and DNN inference |
| `scipy` | Dryden transfer functions, signal processing |
| `scikit-learn` | Scaler for EMPA noise model |
| `bayesian-optimization` | Bayesian Optimization backend |
| `pyyaml` | YAML config parsing |
| `stable-baselines3` | SAC and TD3 reinforcement-learning agents |
| `gymnasium` | RL environment interface |
| `mosqito` | Psychoacoustic metric computation |

---

## 4. Quick Start

```bash
# Run the default simulation (no optimization)
python main.py

# Run a single optimizer
python pid_optimization_bayopt.py

# Launch all optimizers concurrently
python optimization_start.py

# Open the world editor GUI
python world_creation_gui.py
```

---

## 5. Quadcopter Dynamic Model

### 5.1 Reference Frames and Notation

The model adopts two right-handed Cartesian frames:

| Symbol | Frame | Description |
|---|---|---|
| $\mathcal{W}$ | World (inertial) | Fixed to the ground; $z$-axis points upward |
| $\mathcal{B}$ | Body | Fixed to the drone center of mass |

State variables:

| Symbol | Description | Units |
|---|---|---|
| $\mathbf{p} = (x,\,y,\,z)^\top$ | Position in $\mathcal{W}$ | m |
| $\dot{\mathbf{p}} = (v_x,\,v_y,\,v_z)^\top$ | Linear velocity in $\mathcal{W}$ | m/s |
| $\boldsymbol{\Theta} = (\phi,\,\theta,\,\psi)^\top$ | Euler angles (roll, pitch, yaw) | rad |
| $\boldsymbol{\omega} = (\dot\phi,\,\dot\theta,\,\dot\psi)^\top$ | Angular velocity | rad/s |
| $\boldsymbol{\Omega} = (\Omega_1,\,\Omega_2,\,\Omega_3,\,\Omega_4)^\top$ | Motor angular velocities | rad/s |

Physical parameters (default values from `simulation_parameters.yaml`):

| Symbol | Parameter | Default Value |
|---|---|---|
| $m$ | Total mass | 5.2 kg |
| $\mathbf{I} = \mathrm{diag}(I_{xx},\,I_{yy},\,I_{zz})$ | Inertia tensor | $(3.8,\,3.8,\,7.1)\times 10^{-3}$ kg·m² |
| $l$ | Arm length (center to rotor) | 0.32 m |
| $\mathbf{C}_d = (C_{d,x},\,C_{d,y},\,C_{d,z})$ | Translational drag coefficients | $(0.1,\,0.1,\,0.15)$ |
| $\mathbf{C}_a = (C_{a,\phi},\,C_{a,\theta},\,C_{a,\psi})$ | Aerodynamic friction coefficients | $(0.1,\,0.1,\,0.15)$ |
| $J_r$ | Rotor inertia | $6\times 10^{-5}$ kg·m² |
| $g$ | Gravitational acceleration | 9.81 m/s² |
| $\rho$ | Air density | 1.225 kg/m³ |

### 5.2 Translational Dynamics

The rotation matrix $\mathbf{R}(\phi,\theta,\psi) = \mathbf{R}_z(\psi)\,\mathbf{R}_y(\theta)\,\mathbf{R}_x(\phi)$ transforms vectors from body to world frame. The translational equations of motion are:

$$
m\,\ddot{\mathbf{p}} \;=\; \mathbf{R}(\phi,\theta,\psi)\begin{pmatrix} 0 \\ 0 \\ T \end{pmatrix} - m\,\mathbf{g} - \mathbf{C}_d\,\dot{\mathbf{p}}
$$

Expanding component-wise:

$$
\ddot{x} = \frac{T}{m}\bigl(\cos\psi\,\sin\theta\,\cos\phi + \sin\psi\,\sin\phi\bigr) - \frac{C_{d,x}}{m}\,\dot{x}
$$

$$
\ddot{y} = \frac{T}{m}\bigl(\sin\psi\,\sin\theta\,\cos\phi - \cos\psi\,\sin\phi\bigr) - \frac{C_{d,y}}{m}\,\dot{y}
$$

$$
\ddot{z} = \frac{T}{m}\,\cos\theta\,\cos\phi - \frac{C_{d,z}}{m}\,\dot{z} - g
$$

where $T = \sum_{i=1}^{4} T_i$ is the total thrust produced by all rotors.

### 5.3 Rotational Dynamics

The Euler equation for a rigid body with gyroscopic coupling from spinning rotors:

$$
\mathbf{I}\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega}\times(\mathbf{I}\,\boldsymbol{\omega}) - \mathbf{C}_a\,\mathrm{sign}(\boldsymbol{\omega})\circ\boldsymbol{\omega}^2
$$

The net rotor angular momentum produces a gyroscopic precession term proportional to $\Omega_r = \Omega_1 - \Omega_2 + \Omega_3 - \Omega_4$. Expanded per axis:

$$
\ddot{\phi} = \frac{1}{I_{xx}}\Bigl[l\,(T_4 - T_2) - C_{a,\phi}\,\mathrm{sign}(\dot\phi)\,\dot\phi^2 - J_r\,\Omega_r\,\dot\theta - (I_{zz}-I_{yy})\,\dot\theta\,\dot\psi\Bigr]
$$

$$
\ddot{\theta} = \frac{1}{I_{yy}}\Bigl[l\,(T_3 - T_1) - C_{a,\theta}\,\mathrm{sign}(\dot\theta)\,\dot\theta^2 + J_r\,\Omega_r\,\dot\phi - (I_{xx}-I_{zz})\,\dot\phi\,\dot\psi\Bigr]
$$

$$
\ddot{\psi} = \frac{1}{I_{zz}}\Bigl[(Q_1 - Q_2 + Q_3 - Q_4) - C_{a,\psi}\,\mathrm{sign}(\dot\psi)\,\dot\psi^2 - (I_{yy}-I_{xx})\,\dot\phi\,\dot\theta\Bigr]
$$

where $T_i$ and $Q_i$ are the thrust and reaction torque of rotor $i$, respectively.

### 5.4 Motor Mixing

The mixer inverts the relation between high-level control inputs $(u_1,\,u_2,\,u_3,\,u_4)$ and individual motor angular velocities $\omega_i$. Using linearized thrust/torque coefficients $b$ and $d$:

$$
\omega_1^2 = \frac{u_1}{4b} - \frac{u_3}{2bl} + \frac{u_4}{4d}, \qquad
\omega_2^2 = \frac{u_1}{4b} - \frac{u_2}{2bl} - \frac{u_4}{4d}
$$

$$
\omega_3^2 = \frac{u_1}{4b} + \frac{u_3}{2bl} + \frac{u_4}{4d}, \qquad
\omega_4^2 = \frac{u_1}{4b} + \frac{u_2}{2bl} - \frac{u_4}{4d}
$$

Each $\omega_i^2$ is clamped to $[0,\;\omega_{\max}^2]$ before taking the square root. RPM values are obtained via $\mathrm{RPM}_i = \omega_i \cdot 60 / (2\pi)$.

### 5.5 Neural-Network Rotor Model (BEMT)

Rather than relying on simplified $T = b\,\omega^2$ and $Q = d\,\omega^2$ relations, each rotor is modelled by a small feedforward neural network (two hidden layers of 16 units each, ReLU activations) trained on Blade Element Momentum Theory data:

$$
f_{\mathrm{NN}} : \mathrm{RPM} \;\longmapsto\; (T,\;Q,\;P,\;C_T,\;C_Q,\;C_P)
$$

| Output | Description |
|---|---|
| $T$ | Thrust [N] |
| $Q$ | Torque [N·m] |
| $P$ | Power [W] |
| $C_T$ | Thrust coefficient |
| $C_Q$ | Torque coefficient |
| $C_P$ | Power coefficient |

Input and output normalization parameters are stored in `Rotor/normalization_params.pth`. The trained model weights are in `Rotor/rotor_model.pth`. Training data is generated from a Python BEMT solver bundled in `Rotor/pybemt/`.

### 5.6 Numerical Integration

The state is advanced in time using the classical **fourth-order Runge–Kutta (RK4)** scheme:

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{\Delta t}{6}\bigl(\mathbf{k}_1 + 2\,\mathbf{k}_2 + 2\,\mathbf{k}_3 + \mathbf{k}_4\bigr)
$$

where

$$
\mathbf{k}_1 = f(\mathbf{x}_n), \quad
\mathbf{k}_2 = f\!\left(\mathbf{x}_n + \tfrac{\Delta t}{2}\,\mathbf{k}_1\right), \quad
\mathbf{k}_3 = f\!\left(\mathbf{x}_n + \tfrac{\Delta t}{2}\,\mathbf{k}_2\right), \quad
\mathbf{k}_4 = f(\mathbf{x}_n + \Delta t\,\mathbf{k}_3)
$$

The state vector $\mathbf{x}$ comprises position, velocity, Euler angles, and angular velocity. Motor RPMs, thrust, torque, and power are held constant during a single integration step and updated between steps via the controller–mixer–rotor pipeline. Euler angles are wrapped to $[-\pi,\,\pi]$ after each step, and angular velocities are clamped to $|\dot\phi|,|\dot\theta|,|\dot\psi|\leq 10$ rad/s for numerical stability.

The default time step is $\Delta t = 0.008$ s, yielding a dynamics sampling rate of 125 Hz. Logged data is recorded every `frame_skip` steps (default 10), giving an effective logging rate of 12.5 Hz.

---

## 6. Cascade PID Control Architecture

### 6.1 PID Law

Each controller implements the standard discrete PID law:

$$
u(t) = K_P\,e(t) + K_I\int_0^t e(\tau)\,d\tau + K_D\,\frac{de}{dt}
$$

where $e(t) = r(t) - y(t)$ is the tracking error between the reference $r$ and measurement $y$.

### 6.2 Cascade Loops

The controller employs a three-stage cascade architecture with six independent PID controllers:

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                    Cascade PID Controller                    │
                    │                                                              │
  Waypoint ──┬──►  │  ┌──────────┐     ┌──────────┐     ┌──────────┐             │
  (x,y,z)    │     │  │ Position │ v*  │  Speed   │ φ*,θ*│ Attitude │  u2,u3     │ ──► Mixer ──► Motors
             │     │  │ PID (x,y)│────►│ PID (h)  │────►│ PID(φ,θ) │────────────►│
             │     │  └──────────┘     └──────────┘     └──────────┘             │
             │     │                                                              │
             │     │  ┌──────────┐     ┌──────────┐                              │
             └──►  │  │ Altitude │ vz* │  VSpeed  │  u1 (thrust)                 │
                   │  │ PID (z)  │────►│ PID (vz) │──────────────────────────────►│
                   │  └──────────┘     └──────────┘                              │
                   │                                                              │
                   │  ┌──────────┐                                               │
                   │  │ Yaw PID  │  u4                                           │
                   │  │  (ψ)     │──────────────────────────────────────────────►│
                   │  └──────────┘                                               │
                   └──────────────────────────────────────────────────────────────┘
```

**Stage 1 — Position → Desired velocity.** Two position PIDs (shared gains `k_pid_pos`) map horizontal position errors $(e_x,\, e_y)$ to desired velocities $(v_x^*,\, v_y^*)$. A separate altitude PID (`k_pid_alt`) maps $e_z$ to a desired vertical speed $v_z^*$.

**Stage 2 — Velocity → Desired angles / thrust.** A horizontal speed PID (`k_pid_hsp`) converts velocity errors to desired roll $\phi^*$ and pitch $\theta^*$ commands (clamped to $\pm\theta_{\max}$). A vertical speed PID (`k_pid_vsp`) produces a thrust adjustment added to the hover-compensated thrust:

$$
u_1 = \underbrace{m\,g\;\min\!\Bigl(\frac{1}{\cos\theta\,\cos\phi},\;1.5\Bigr)}_{\text{hover compensation}} + \;\Delta u_{v_z}
$$

**Stage 3 — Attitude tracking.** Two attitude PIDs (shared gains `k_pid_att`) track the desired roll and pitch. A dedicated yaw PID (`k_pid_yaw`) stabilises the heading.

The six gain sets optimized during tuning are:

| Gain Set | Controller | Notation |
|---|---|---|
| `k_pid_pos` | Horizontal position (x, y) | $(K_P^{\mathrm{pos}},\;K_I^{\mathrm{pos}},\;K_D^{\mathrm{pos}})$ |
| `k_pid_alt` | Altitude (z) | $(K_P^{\mathrm{alt}},\;K_I^{\mathrm{alt}},\;K_D^{\mathrm{alt}})$ |
| `k_pid_att` | Attitude (roll, pitch) | $(K_P^{\mathrm{att}},\;K_I^{\mathrm{att}},\;K_D^{\mathrm{att}})$ |
| `k_pid_yaw` | Yaw | $(K_P^{\mathrm{yaw}},\;K_I^{\mathrm{yaw}},\;K_D^{\mathrm{yaw}})$ — kept fixed |
| `k_pid_hsp` | Horizontal speed | $(K_P^{\mathrm{hsp}},\;K_I^{\mathrm{hsp}},\;K_D^{\mathrm{hsp}})$ |
| `k_pid_vsp` | Vertical speed | $(K_P^{\mathrm{vsp}},\;K_I^{\mathrm{vsp}},\;K_D^{\mathrm{vsp}})$ |

The yaw gains are fixed at $(0.5,\;10^{-6},\;0.1)$ during all optimizations; the remaining 15 scalar gains form the optimizable parameter vector $\boldsymbol{\vartheta}\in\mathbb{R}^{15}$.

### 6.3 Anti-Windup and Saturation

Each PID integrator is clamped to prevent windup:

$$
\bigl|\textstyle\int e\,dt\bigr| \;\leq\; \alpha \cdot U_{\max}
$$

where $\alpha$ is the `anti_windup_contrib` factor (default 0.4) and $U_{\max}$ is the saturation limit of the corresponding control output. The final outputs are clipped to their respective physical limits (e.g., thrust $\in [0,\, T_{\max}]$; roll, pitch, yaw commands $\in [-U_{\max},\, U_{\max}]$).

Desired horizontal speed is further clamped so that $\|\mathbf{v}_h^*\| \leq v_{h,\max}$ (default 50 km/h) and vertical speed to $|v_z^*| \leq v_{z,\max}$ (default 20 km/h).

---

## 7. Dryden Wind Turbulence Model

Atmospheric turbulence is generated using the **Dryden spectral model** (MIL-HDBK-1797). For each axis $u$ (longitudinal), $v$ (lateral), and $w$ (vertical), a continuous-time transfer function shapes white noise into a spatially correlated wind disturbance.

**Longitudinal component ($u$-axis):**

$$
G_u(s) = \sigma_u\,\sqrt{\frac{2\,L_u}{\pi\,V}}\;\cdot\;\frac{V}{L_u\,s + V}
$$

**Lateral and vertical components ($v$, $w$-axes):**

$$
G_{v,w}(s) = \sigma\,\sqrt{\frac{L}{\pi\,V}}\;\cdot\;\frac{\sqrt{3}\,\frac{L}{V}\,s + 1}{\left(\frac{L}{V}\right)^2 s^2 + 2\,\frac{L}{V}\,s + 1}
$$

where:

| Symbol | Description |
|---|---|
| $V$ | Airspeed of the vehicle [m/s] |
| $h$ | Height above ground [m] |
| $L = h\,/\,(0.177 + 0.000823\,h)^{0.2}$ | Turbulence scale length [m] |
| $\sigma = 0.1 \cdot \mathrm{turb\_level} \,/\, (0.177 + 0.000823\,h)^{0.4}$ | Turbulence intensity [m/s] |
| $\mathrm{turb\_level}$ | Configurable turbulence intensity parameter |

White noise is filtered through these transfer functions using `scipy.signal.lsim` to produce time-domain wind velocity histories. The resulting wind components are injected into the rotor thrust equation:

$$
\Delta T_i = \frac{\rho}{2}\Bigl[2\pi^2\,V_z\,\Omega_i\,(R_{\mathrm{tip}}^2 - R_{\mathrm{root}}^2) + \frac{2\pi^2\,V_z}{\Omega_i}\,(V_x^2+V_y^2)\,\ln\!\frac{R_{\mathrm{tip}}}{R_{\mathrm{root}}}\Bigr]
$$

where $R_{\mathrm{tip}}$ and $R_{\mathrm{root}}$ are the rotor tip and root radii respectively.

---

## 8. Acoustic Noise Model

Two interchangeable noise models are provided. Both predict per-band Sound Power Level (SWL) and Sound Pressure Level (SPL) in 1/3-octave bands.

### 8.1 Lookup-Table Sound Model (DNN)

The primary model uses a pre-computed lookup table (LUT) of shape $(n_{\mathrm{angles}},\, n_{\mathrm{bands}})$ that maps radiation angle $\zeta$ to reference SWL spectra. The procedure is:

1. **Angle indexing.** Convert the elevation angle $\zeta = \arctan(|z_d| / d_{\mathrm{horiz}})$ to a discrete LUT row index.

2. **Single-rotor reference.** The LUT stores total-drone levels, converted to a single-rotor reference:

$$
L_{w,\mathrm{ref}}^{(1)} = L_{w,\mathrm{ref}}^{(N)} - 10\,\log_{10}(N_{\mathrm{rotors}})
$$

3. **RPM scaling.** For each rotor $i$ spinning at $\mathrm{RPM}_i$:

$$
L_{w,i}(f) = L_{w,\mathrm{ref}}^{(1)}(f) + 10\,\beta\,\log_{10}\!\left(\frac{\mathrm{RPM}_i}{\mathrm{RPM}_{\mathrm{ref}}}\right)
$$

where $\beta = 3$ is the RPM exponent (default).

4. **Power summation.** Total SWL per band:

$$
L_{w,\mathrm{tot}}(f) = 10\,\log_{10}\!\sum_{i=1}^{N_{\mathrm{rotors}}} 10^{L_{w,i}(f)/10}
$$

### 8.2 EMPA Regression Model

An alternative parametric model based on the EMPA approach uses regression coefficients $(a,\,b,\,c,\,d)$ fitted per frequency band:

$$
L_{w,i}(f) = L_{w,\mathrm{ref}}(f) + a(f)\,\zeta^2 + b(f)\,|\zeta| + c(f)\,\Omega_i + d(f)\,\Omega_i^2 + C_{\mathrm{proc}} - 10\,\log_{10}(N_{\mathrm{rotors}})
$$

The total SWL is then obtained by power-summing over all rotors as above.

### 8.3 Propagation and Atmospheric Absorption

SPL at a receiver located at distance $r$ from the source is computed per band:

$$
L_p(f) = L_w(f) - 10\,\log_{10}(4\pi r^2) - \alpha(f)\,r + \mathrm{DI}(f)
$$

where:

- $\alpha(f)$ is the atmospheric absorption coefficient in dB/m, computed per ISO 9613-1 from ambient temperature $T$, relative humidity $\mathrm{RH}$, and static pressure $P$.
- $\mathrm{DI}(f)$ is the directivity index (optional, defaults to 0 dB — omnidirectional).

Broadband levels are obtained by power-summing all bands:

$$
L_{\mathrm{broadband}} = 10\,\log_{10}\!\sum_{k=1}^{n_{\mathrm{bands}}} 10^{L_k/10}
$$

---

## 9. Psychoacoustic Annoyance

When enabled, the simulation computes **Zwicker's psychoacoustic annoyance** (PA) for each ground-level grid cell that falls within the drone's noise annoyance radius. The procedure uses the [MoSQITo](https://github.com/Eomys/MoSQITo) library and proceeds as follows:

1. **Loudness** $N(t)$ — Time-varying specific loudness is computed per ISO 532-1 (Zwicker stationary method applied frame-by-frame). The time-averaged loudness is $\bar{N}$, and the 95th-percentile loudness (N5) is:

$$
N_5 = \mathrm{percentile}_{95}\bigl(N(t)\bigr) \quad [\mathrm{sone}]
$$

2. **Sharpness** $S$ — Computed from the specific loudness pattern per DIN 45692:

$$
S \quad [\mathrm{acum}]
$$

3. **Roughness** $R$ — Computed per Daniel & Weber from the modulation spectrum:

$$
R \quad [\mathrm{asper}]
$$

4. **Fluctuation Strength** $F_S$ — Computed per Aures/Daniel & Weber:

$$
F_S \quad [\mathrm{vacil}]
$$

5. **Psychoacoustic Annoyance** — The Zwicker PA metric combines the above:

$$
\omega_S = \max(S - 1.75,\;0)\;\ln(N_5 + 10)
$$

$$
\omega_{FR} = \frac{2.18}{N_5^{0.4}}\;(0.4\,F_S + 0.6\,R)
$$

$$
\mathrm{PA} = N_5\;\Bigl(1 + \sqrt{\omega_S^2 + \omega_{FR}^2}\Bigr)
$$

---

## 10. World and Environment Model

The simulation takes place in a grid-based world (class `World`), where the environment is partitioned into square cells of configurable side length. Each cell is assigned an **area type**:

| ID | Area Type | Min Altitude | Noise Penalty |
|----|-----------|-------------|---------------|
| 1 | Housing Estate | 150 m | 3 |
| 2 | Industrial Area | 70 m | 1 |
| 3 | Open Field | 0 m | 0 |
| 4 | Forbidden Area | — | 10 |

Worlds are serialized as pickle files and stored in the `Worlds/` directory. A GUI editor (`world_creation_gui.py`) allows interactive creation and editing. An optional background image can be loaded for visual reference.

---

## 11. Cost Function for Optimization

The optimization objective is a composite scalar cost $J(\boldsymbol{\vartheta})$ that penalizes poor flight performance, excessive control activity, energy consumption, and noise impact:

$$
J = J_{\mathrm{time}} + J_{\mathrm{dist}} + J_{\mathrm{osc}} + J_{\mathrm{compl}} + J_{\mathrm{over}} + J_{\mathrm{power}} + J_{\mathrm{noise}} + J_{\mathrm{stall}}
$$

Each component is defined as follows:

| Term | Definition | Weight |
|---|---|---|
| $J_{\mathrm{time}}$ | Navigation time $t_f$ [s] | 1 |
| $J_{\mathrm{dist}}$ | Final distance to last waypoint: $d_f^{0.9}$ | 1 |
| $J_{\mathrm{osc}}$ | $w_{\mathrm{osc}}\bigl(w_{\phi\theta}\sum\lvert\Delta\phi_k\rvert + w_{\phi\theta}\sum\lvert\Delta\theta_k\rvert + w_T\sum\lvert\Delta T_k\rvert\bigr)$ | $w_{\mathrm{osc}}=1,\ w_{\phi\theta}=1,\ w_T=10^{-5}$ |
| $J_{\mathrm{compl}}$ | $w_c\,(1 - n_{\mathrm{completed}}/n_{\mathrm{total}})$ | $w_c = 1000$ |
| $J_{\mathrm{over}}$ | $w_o \sum_{\mathrm{seg}} \int \max(d(t) - d_{\mathrm{shift}},\, 0)\,dt$ | $w_o = 0.02$ |
| $J_{\mathrm{power}}$ | $w_P \sum P_k$ | $w_P = 10^{-4}$ |
| $J_{\mathrm{noise}}$ | $w_n\bigl(\lVert\mathbf{L}_w\rVert_p^p + \max L_w\bigr)$, with $p=12$ | $w_n = 2\times10^{-25}$ |
| $J_{\mathrm{stall}}$ | Penalty of $+1000$ if drone is detected as not moving | — |

All weights are configurable. The cost function is implemented in `opt_func.py::calculate_costs()`.

---

## 12. Optimization Algorithms

All optimizers inherit from a common `Optimizer` base class that handles configuration loading, world/noise model initialization, waypoint generation, and logging. The optimizable vector $\boldsymbol{\vartheta}\in\mathbb{R}^{15}$ collects the $(K_P,\,K_I,\,K_D)$ gains for Position, Altitude, Attitude, Horizontal Speed, and Vertical Speed controllers (yaw is fixed).

Each optimizer supports the following common options:

| Option | Description | Default |
|---|---|---|
| `set_initial_obs` | Seed the search with current PID gains | `True` |
| `simulate_wind_flag` | Enable Dryden turbulence during evaluation | `False` |
| `study_name` | Suffix for the output folder | `""` |
| `simulation_time` | Max simulation duration per evaluation [s] | `150` |

### 12.1 Bayesian Optimization

Surrogate-based sequential optimization using a Gaussian Process (GP) model. An acquisition function (Upper Confidence Bound by default) selects the next evaluation point.

- **Library:** `bayesian-optimization`
- **Config file:** `Settings/bay_opt.yaml`
- **Script:** `pid_optimization_bayopt.py`
- **Key hyperparameters:** `n_iter` (iterations), `init_points` (random initial probes)

The objective is the negative of the total cost ($-J$) since the library maximizes.

### 12.2 Particle Swarm Optimization (PSO)

Population-based stochastic optimizer. Each particle maintains a position and velocity in $\mathbb{R}^{15}$:

$$
\mathbf{v}_i^{(t+1)} = w\,\mathbf{v}_i^{(t)} + c_1\,\mathbf{r}_1\circ(\mathbf{p}_i^{\mathrm{best}} - \mathbf{x}_i^{(t)}) + c_2\,\mathbf{r}_2\circ(\mathbf{g}^{\mathrm{best}} - \mathbf{x}_i^{(t)})
$$

$$
\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}
$$

where $w$ is the inertia weight, $c_1$ and $c_2$ are the cognitive and social coefficients, and $\mathbf{r}_1,\,\mathbf{r}_2$ are uniform random vectors.

- **Config file:** `Settings/pso_opt.yaml`
- **Script:** `pid_optimization_pso.py`
- **Key hyperparameters:** `swarm_size` (30), `inertia_weight` (0.7), `cognitive_coeff` (1.5), `social_coeff` (1.5), `n_iter` (100)

### 12.3 Genetic Algorithm (GA)

Evolutionary optimizer with tournament selection, single-point crossover, and random-reset mutation:

1. **Selection:** Tournament of size $k$ — pick the fittest of $k$ random individuals.
2. **Crossover:** With probability $p_c$, swap genes at a random crossover point.
3. **Mutation:** Each gene mutates independently with probability $p_m$ by resampling uniformly within bounds.
4. **Elitism:** The top $\lfloor\epsilon\cdot N_{\mathrm{pop}}\rfloor$ individuals are preserved.

- **Config file:** `Settings/ga_opt.yaml`
- **Script:** `pid_optimization_ga.py`
- **Key hyperparameters:** `population_size` (30), `n_generations` (100), `crossover_rate` (0.8), `mutation_rate` (0.1), `tournament_size` (3), `elite_fraction` (0.1)

### 12.4 Grey Wolf Optimizer (GWO)

A nature-inspired metaheuristic that models the social hierarchy of grey wolves (alpha $\alpha$, beta $\beta$, delta $\delta$, omega $\omega$). Wolves update their positions guided by the three best solutions:

$$
\mathbf{x}_i^{(t+1)} = \frac{1}{3}\bigl(\mathbf{X}_1 + \mathbf{X}_2 + \mathbf{X}_3\bigr)
$$

where

$$
\mathbf{X}_1 = \boldsymbol{\alpha} - \mathbf{A}_1\,|\mathbf{C}_1\,\boldsymbol{\alpha} - \mathbf{x}_i|, \quad
\mathbf{X}_2 = \boldsymbol{\beta} - \mathbf{A}_2\,|\mathbf{C}_2\,\boldsymbol{\beta} - \mathbf{x}_i|, \quad
\mathbf{X}_3 = \boldsymbol{\delta} - \mathbf{A}_3\,|\mathbf{C}_3\,\boldsymbol{\delta} - \mathbf{x}_i|
$$

The parameter $a$ decays linearly from 2 to 0 over iterations, controlling the exploration–exploitation trade-off: $\mathbf{A} = 2a\,\mathbf{r}_1 - a$ and $\mathbf{C} = 2\,\mathbf{r}_2$.

- **Config file:** `Settings/gwo_opt.yaml`
- **Script:** `pid_optimization_gwo.py`
- **Key hyperparameters:** `pack_size` (30), `n_iter` (100)

### 12.5 Soft Actor–Critic (SAC)

An off-policy maximum-entropy reinforcement learning algorithm. The PID tuning problem is cast as a **one-step episodic MDP** with a custom Gymnasium environment:

- **Observation space:** $\boldsymbol{\vartheta}\in\mathbb{R}^{15}$, bounded by gain limits.
- **Action space:** $\boldsymbol{\vartheta}\in\mathbb{R}^{15}$, continuous, bounded.
- **Reward:** $r = -J(\boldsymbol{\vartheta})$
- **Episode:** Single step (terminated immediately after one simulation).

SAC maximizes the entropy-regularized expected return:

$$
\pi^* = \arg\max_\pi \;\mathbb{E}_\pi\!\Bigl[\sum_t r_t + \alpha\,\mathcal{H}\bigl(\pi(\cdot|s_t)\bigr)\Bigr]
$$

Random pre-fill is disabled (`learning_starts=0`) so the first attempt uses the initial PID gains when `set_initial_obs=True`.

- **Library:** `stable-baselines3`
- **Config file:** `Settings/sac_opt.yaml`
- **Script:** `pid_optimization_sac.py`
- **Key hyperparameters:** `total_timesteps` (3000)

### 12.6 Twin-Delayed DDPG (TD3)

A deterministic policy gradient algorithm with three improvements over DDPG: clipped double Q-learning, delayed policy updates, and target policy smoothing. The MDP formulation is identical to SAC. Gaussian exploration noise $\mathcal{N}(0,\,\sigma^2\mathbf{I})$ is added to actions.

- **Library:** `stable-baselines3`
- **Config file:** `Settings/td3_opt.yaml`
- **Script:** `pid_optimization_td3.py`
- **Key hyperparameters:** `total_timesteps` (3000), `action_noise_sigma` (0.1)

---

## 13. Configuration Reference

### `Settings/simulation_parameters.yaml`

| Parameter | Type | Description |
|---|---|---|
| `dt` | float | Physics time step [s] |
| `simulation_time` | float | Maximum simulation duration [s] |
| `frame_skip` | int | Logging decimation factor |
| `threshold` | float | Distance to consider a waypoint reached [m] |
| `target_shift_threshold_distance` | float | Distance to advance to next segment [m] |
| `noise_annoyance_radius` | int | Radius for noise emission map [m] |
| `max_rpm` | float | Maximum motor RPM |
| `n_rotors` | int | Number of rotors |
| `max_roll_angle`, `max_pitch_angle`, `max_yaw_angle` | float | Command saturation limits [deg] |
| `max_h_speed_limit_kmh` | float | Maximum horizontal speed [km/h] |
| `max_v_speed_limit_kmh` | float | Maximum vertical speed [km/h] |
| `max_angle_limit_deg` | float | Maximum roll/pitch command [deg] |
| `anti_windup_contrib` | float | Anti-windup integral limit fraction |
| `m` | float | Drone mass [kg] |
| `I` | list[3] | Diagonal inertia tensor [kg·m²] |
| `d` | float | Drag factor |
| `l` | float | Arm length [m] |
| `Cd` | list[3] | Translational drag coefficients |
| `Ca` | list[3] | Aerodynamic friction coefficients |
| `Jr` | float | Rotor inertia [kg·m²] |
| `k_pid_*` | list[3] | PID gains $(K_P,\, K_I,\, K_D)$ |
| `world_data_path` | str | Path to serialized world file |
| `rotor_model_path` | str | Path to PyTorch rotor model |
| `norm_params_path` | str | Path to normalization parameters |
| `dnn_model_filename` | str | Path to DNN noise lookup table |
| `empa_model_filename` | str | Path to EMPA model coefficients |
| `empa_model_scaling_filename` | str | Path to EMPA scaler |
| `rpm_reference` | float | Reference RPM for noise model |

### Optimizer YAML files

Each optimizer YAML (`bay_opt.yaml`, `pso_opt.yaml`, `ga_opt.yaml`, `gwo_opt.yaml`, `sac_opt.yaml`, `td3_opt.yaml`) contains:

- Algorithm-specific hyperparameters (iterations, population size, learning rates, etc.)
- `pbounds`: A dictionary mapping each of the 15 gain names to `[lower_bound, upper_bound]`

---

## 14. Project Structure

```
├── main.py                      # Simulation entry point
├── Controller.py                # PID cascade controller
├── Drone.py                     # 6-DOF quadcopter dynamics
├── Simulation.py                # Simulation loop, wind injection, noise computation
├── Wind.py                      # Dryden turbulence transfer functions
├── World.py                     # Grid-based environment model
├── world_creation_gui.py        # GUI for creating/editing worlds
├── opt_func.py                  # Cost function, simulation runner, logging utilities
├── optimizer.py                 # Base optimizer class
├── optimization_start.py        # Launch all optimizers concurrently
├── pid_optimization_bayopt.py   # Bayesian Optimization
├── pid_optimization_pso.py      # Particle Swarm Optimization
├── pid_optimization_ga.py       # Genetic Algorithm
├── pid_optimization_gwo.py      # Grey Wolf Optimizer
├── pid_optimization_sac.py      # Soft Actor–Critic (RL)
├── pid_optimization_td3.py      # Twin-Delayed DDPG (RL)
├── plotting_functions.py        # Visualization: animation, logs, noise maps
├── utils.py                     # Angle wrapping, rotation matrices
├── Drone_Legacy.py              # Legacy drone model (deprecated)
├── requirements.txt             # Python dependencies
│
├── Noise/                       # Acoustic emission models
│   ├── DNNModel.py              # Angle/RPM lookup-table noise model
│   ├── EmpaModel.py             # EMPA regression noise model
│   ├── AIModel.py               # DNN-based noise model (alternative)
│   ├── Psychoacoustic.py        # MoSQITo-backed psychoacoustic backend
│   ├── lookup_noise_model.npy   # Pre-computed SWL lookup table
│   ├── model_coefficients.npz   # EMPA model coefficients
│   ├── scaler.joblib            # EMPA scaler
│   ├── csv/                     # Training data for noise models
│   └── output/                  # Noise model outputs
│
├── Rotor/                       # Neural-network rotor model
│   ├── TorchRotorModel.py       # PyTorch rotor DNN
│   ├── rotor_model.pth          # Trained model weights
│   ├── normalization_params.pth # Input/output normalization
│   ├── rotor_config.ini         # Rotor geometry configuration
│   ├── rotor_model_data_bemt.csv# BEMT training data
│   ├── rotor_model_creator.ipynb# Model training notebook
│   └── pybemt/                  # Python BEMT solver
│
├── Settings/                    # YAML configuration files
│   ├── simulation_parameters.yaml
│   ├── bay_opt.yaml
│   ├── pso_opt.yaml
│   ├── ga_opt.yaml
│   ├── gwo_opt.yaml
│   ├── sac_opt.yaml
│   └── td3_opt.yaml
│
├── Worlds/                      # Serialized world files (.pkl)
├── Optimizations/               # Optimization results (per algorithm, timestamped)
├── Plots/                       # Generated plots
└── Pic/                         # README assets
```

---

## 15. Usage Guide

### 15.1 Running a Simulation

```bash
python main.py
```

This executes a full simulation with the PID gains specified in `Settings/simulation_parameters.yaml`, produces a 3D trajectory animation, time-series log plots, noise emission maps with psychoacoustic analysis, and prints cost metrics to the console.

**Customising waypoints:** Edit the `create_training_waypoints()` function in `main.py`, or use `create_random_waypoints(n, x_range, y_range, z_range, v)` for random waypoints.

**Enabling wind:** Uncomment the `sim.setWind(...)` line in `main.py`.

**Switching noise model:** Replace `load_dnn_noise_model(parameters)` with `load_empa_noise_model(parameters)`.

### 15.2 Running a Single Optimizer

Each optimizer can be launched independently:

```bash
python pid_optimization_bayopt.py   # Bayesian Optimization
python pid_optimization_pso.py      # Particle Swarm Optimization
python pid_optimization_ga.py       # Genetic Algorithm
python pid_optimization_gwo.py      # Grey Wolf Optimizer
python pid_optimization_sac.py      # Soft Actor–Critic
python pid_optimization_td3.py      # Twin-Delayed DDPG
```

To customize optimizer settings, edit the corresponding YAML file in `Settings/` or pass arguments to the optimizer constructor in the script's `main()` function:

```python
optimizer = BayesianPIDOptimizer(
    config_file="Settings/bay_opt.yaml",
    parameters_file="Settings/simulation_parameters.yaml",
    verbose=True,
    set_initial_obs=True,        # Seed with current gains
    simulate_wind_flag=True,     # Enable turbulence
    study_name="my_experiment",  # Output folder suffix
    simulation_time=150,         # Max sim time per eval
    waypoints=my_custom_waypoints,
)
optimizer.optimize()
```

### 15.3 Running All Optimizers Concurrently

```bash
python optimization_start.py
```

This launches all six optimizers as separate processes. Each process writes to its own timestamped folder under `Optimizations/<METHOD>/`.

### 15.4 World Creation GUI

```bash
python world_creation_gui.py
```

Interactively define area types (Housing, Industrial, Open Field, Forbidden) on a grid and save/load world files used by the simulation.

---

## 16. Output and Results

Each optimization run creates a timestamped directory:

```
Optimizations/<METHOD>/<YYYYMMDD_HHMMSS_study_name>/
├── best_parameters.txt       # Best PID gains, cost value, runtime summary
├── optimization_log.json     # Per-step JSON log (params, cost, timing)
├── best_parameters_costs.png # Cost trend plot (all evaluations)
└── best_parameters_best_costs.png  # Best-so-far cost trend plot
```

The JSON log contains one entry per line with the following structure:

```json
{
  "target": -42.1234,
  "params": {
    "k_pid_pos": [1.12, 0.0, 0.14],
    "k_pid_alt": [1.19, 0.0, 0.016]
  },
  "datetime": {
    "datetime": "2025-08-20 11:52:25",
    "elapsed": 123.45,
    "delta": 2.31
  },
  "costs": {
    "total_cost": 42.1234,
    "time_cost": 25.3,
    "final_distance_cost": 1.2,
    "oscillation_cost": 5.6,
    "completition_cost": 0.0,
    "overshoot_cost": 3.1,
    "power_cost": 4.5,
    "noise_cost": 1.2
  }
}
```

To replay the best gains found, copy the optimal values from `best_parameters.txt` into `Settings/simulation_parameters.yaml` and run `python main.py`.

---

## 17. References

1. Sabir, A., *et al.* "Modeling of a Quadcopter Trajectory Tracking System Using PID Controller." (2020).
2. MIL-HDBK-1797, *Flying Qualities of Piloted Aircraft*, U.S. Department of Defense.
3. ISO 9613-1:1993, *Acoustics — Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere*.
4. ISO 532-1:2017, *Acoustics — Methods for calculating loudness — Part 1: Zwicker method*.
5. DIN 45692:2009, *Measurement technique for the simulation of the auditory sensation of sharpness*.
6. Daniel, P. and Weber, R. "Psychoacoustical Roughness: Implementation of an Optimized Model." *Acustica*, 83(1), 1997.
7. Zwicker, E. and Fastl, H. *Psychoacoustics: Facts and Models*. Springer, 3rd edition, 2007.
8. Snoek, J., Larochelle, H., and Adams, R.P. "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS*, 2012.
9. Kennedy, J. and Eberhart, R. "Particle Swarm Optimization." *Proc. IEEE ICNN*, 1995.
10. Mirjalili, S., Mirjalili, S.M., and Lewis, A. "Grey Wolf Optimizer." *Advances in Engineering Software*, 69, 2014.
11. Haarnoja, T., *et al.* "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML*, 2018.
12. Fujimoto, S., Hoof, H., and Meger, D. "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*, 2018.
