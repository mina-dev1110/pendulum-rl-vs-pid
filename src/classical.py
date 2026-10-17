"""Classical controller for Gymnasium's Pendulum-v1: energy-based swing-up + PID stabilisation.

Conventions (Pendulum-v1): theta = 0 is the UPRIGHT position, torque is limited to +-2 N.m,
which is too weak to lift the pendulum directly, so it must swing up first.
Dynamics: theta_ddot = (3 g / 2 l) sin(theta) + (3 / m l^2) u
"""
import numpy as np


class SwingUpPID:
    def __init__(self, kp=12.0, ki=0.0, kd=2.5, k_energy=2.0, switch_angle=0.5,
                 m=1.0, l=1.0, g=10.0, dt=0.05, max_torque=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.k_energy = k_energy
        self.switch_angle = switch_angle
        self.m, self.l, self.g, self.dt = m, l, g, dt
        self.max_torque = max_torque
        self.integral = 0.0
        self.mode = "swing"

    def reset(self):
        self.integral = 0.0
        self.mode = "swing"

    def energy_error(self, theta, theta_dot):
        inertia = self.m * self.l**2 / 3.0
        energy = 0.5 * inertia * theta_dot**2 + self.m * self.g * (self.l / 2.0) * np.cos(theta)
        energy_top = self.m * self.g * (self.l / 2.0)  # energy at theta = 0 with zero speed
        return energy - energy_top

    def act(self, obs):
        cos_t, sin_t, theta_dot = obs
        theta = np.arctan2(sin_t, cos_t)  # wrapped to [-pi, pi]

        if abs(theta) < self.switch_angle and abs(theta_dot) < 4.0:
            self.mode = "pid"
        elif abs(theta) > 1.0:
            self.mode = "swing"
            self.integral = 0.0

        if self.mode == "pid":
            self.integral = np.clip(self.integral + theta * self.dt, -1.0, 1.0)
            u = -(self.kp * theta + self.ki * self.integral + self.kd * theta_dot)
        else:
            # pump energy in the direction of motion until the top energy is reached
            # power injected by the torque is u * theta_dot, so push along the motion
            direction = np.sign(theta_dot)
            if direction == 0:
                direction = 1.0
            u = -self.k_energy * self.energy_error(theta, theta_dot) * direction

        return np.clip(np.array([u], dtype=np.float32), -self.max_torque, self.max_torque)
