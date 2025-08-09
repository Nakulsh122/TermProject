from vpython import *
import numpy as np
import time
from math import sin, cos, radians

# ==== Fake IMU Data Generator ====
def generate_fake_imu_data():
    t = 0.0
    dt = 0.05  # 20 Hz
    while True:
        gx = 30 * np.sin(t / 3)   # deg/s
        gy = 20 * np.sin(t / 4)
        gz = 15 * np.sin(t / 5)

        ax = np.sin(t / 6) * 0.2  # g's
        ay = np.sin(t / 8) * 0.2
        az = 0.0  # ignoring gravity for simulated movement

        yield t, ax, ay, az, gx, gy, gz
        t += dt
        time.sleep(dt)

# ==== Complementary Filter ====
def complementary_filter(accel, gyro, dt, prev_angles):
    alpha = 0.98
    accel_pitch = np.arctan2(accel[1], 1.0) * 180 / np.pi
    accel_roll = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + 1.0)) * 180 / np.pi

    gyro_pitch = prev_angles[0] + gyro[0] * dt
    gyro_roll = prev_angles[1] + gyro[1] * dt
    gyro_yaw = prev_angles[2] + gyro[2] * dt

    pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch
    roll = alpha * gyro_roll + (1 - alpha) * accel_roll
    yaw = gyro_yaw
    return pitch, roll, yaw

# ==== Convert Euler angles to rotation matrix ====
def euler_to_matrix(pitch, roll, yaw):
    pitch = radians(pitch)
    roll = radians(roll)
    yaw = radians(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cos(roll), -sin(roll)],
                   [0, sin(roll), cos(roll)]])
    Ry = np.array([[cos(pitch), 0, sin(pitch)],
                   [0, 1, 0],
                   [-sin(pitch), 0, cos(pitch)]])
    Rz = np.array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# ==== Scenes ====
scene1 = canvas(title="IMU Orientation", width=800, height=600, background=color.white)
scene2 = canvas(title="Position Tracking", width=800, height=600, background=color.white)

# ==== Orientation Visual (scene1) ====
imu_box = box(canvas=scene1, length=4, height=0.2, width=2, color=color.orange)
arrow(canvas=scene1, pos=vector(0, 0, 0), axis=vector(3, 0, 0), color=color.red, shaftwidth=0.02)
arrow(canvas=scene1, pos=vector(0, 0, 0), axis=vector(0, 3, 0), color=color.green, shaftwidth=0.02)
arrow(canvas=scene1, pos=vector(0, 0, 0), axis=vector(0, 0, 3), color=color.blue, shaftwidth=0.02)

# ==== Position Tracking Visual (scene2) ====
floor = box(canvas=scene2, pos=vector(0, -0.1, 0), length=20, height=0.1, width=20, color=color.gray(0.8))
tracker = sphere(canvas=scene2, pos=vector(0, 0, 0), radius=0.2, color=color.cyan, make_trail=True)

# ==== Graph ====
graph(title="Pitch, Roll, Yaw over Time", xtitle="Time (s)", ytitle="Angle (°)", fast=False)
pitch_curve = gcurve(color=color.red, label="Pitch")
roll_curve = gcurve(color=color.green, label="Roll")
yaw_curve = gcurve(color=color.blue, label="Yaw")

# ==== Main Loop ====
imu_data = generate_fake_imu_data()
current_angles = [0, 0, 0]  # pitch, roll, yaw

velocity = vector(0, 0, 0)
position = vector(0, 0, 0)
dt = 0.05

for t, ax_, ay_, az_, gx_, gy_, gz_ in imu_data:
    # ---- Orientation ----
    current_angles = complementary_filter([ax_, ay_, az_], [gx_, gy_, gz_], dt, current_angles)
    pitch, roll, yaw = current_angles

    R = euler_to_matrix(pitch, roll, yaw)
    forward = vector(*R[:, 0])
    up_dir = vector(*R[:, 2])
    imu_box.axis = forward
    imu_box.up = up_dir

    pitch_curve.plot(t, pitch)
    roll_curve.plot(t, roll)
    yaw_curve.plot(t, yaw)

    # ---- Position Tracking ----
    accel_vector = vector(ax_, ay_, az_) * 9.81  # convert g's to m/s²
    velocity += accel_vector * dt
    position += velocity * dt
    tracker.pos = position

    rate(20)
