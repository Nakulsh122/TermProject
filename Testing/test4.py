import csv
import time
import numpy as np
from vpython import canvas, box, vector, rate, sphere
import matplotlib.pyplot as plt

CSV_FILE = r"d:\SEM7\Ubiqutous\TermProject\Testing\test2.csv"


# ---------- VPython setup ----------
scene1 = canvas(title="3D Orientation + Position", width=600, height=400)
scene1.forward = vector(-1, -1, -1)
device_box = box(length=1, height=0.2, width=0.5, color=vector(0, 0.6, 1))
pos_sphere = sphere(radius=0.05, color=vector(1, 0, 0), make_trail=True)

# ---------- Matplotlib setup ----------
plt.ion()
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlim(-2, 2)
ax3d.set_ylim(-2, 2)
ax3d.set_zlim(-2, 2)
line3d, = ax3d.plot([], [], [], 'r-')  # trajectory
pos_data = []

# ---------- State variables ----------
orientation = np.eye(3)  # rotation matrix
velocity = np.zeros(3)
position = np.zeros(3)
prev_time = None

# ---------- CSV Reading ----------
with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Get sensor values
        t = float(row["time"])
        ax, ay, az = float(row["ax"]), float(row["ay"]), float(row["az"])
        wx, wy, wz = float(row["wx"]), float(row["wy"]), float(row["wz"])

        if prev_time is None:
            prev_time = t
            continue

        dt = t - prev_time
        prev_time = t

        # --- Update orientation using gyro ---
        omega = np.array([wx, wy, wz])  # rad/s
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        orientation = orientation @ (np.eye(3) + omega_skew * dt)

        # --- Rotate acceleration to world frame ---
        acc_world = orientation @ np.array([ax, ay, az])
        acc_world[2] -= 9.81  # remove gravity (m/s²)

        # --- Integrate for velocity & position ---
        velocity += acc_world * dt
        position += velocity * dt

        # --- Update VPython ---
        device_box.axis = vector(*orientation[:, 0])
        device_box.up = vector(*orientation[:, 2])
        pos_sphere.pos = vector(*position)

        # --- Update Matplotlib ---
        pos_data.append(position.copy())
        xs, ys, zs = zip(*pos_data)
        line3d.set_data(xs, ys)
        line3d.set_3d_properties(zs)
        plt.draw()
        plt.pause(0.001)

        rate(60)  # VPython render speed

plt.ioff()
plt.show()
