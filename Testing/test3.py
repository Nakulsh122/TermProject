from vpython import *
import csv
import numpy as np

# === SETTINGS ===
CSV_FILE = r"d:\SEM7\Ubiqutous\TermProject\Testing\test2.csv"
  # change to your CSV file
g = 9.81

# === FUNCTIONS ===
def axis_angle_to_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])

# === SCENE SETUP ===
scene1 = canvas(title="Orientation Tracking", width=600, height=400, center=vector(0,0,0))
scene2 = canvas(title="Position Tracking", width=600, height=400, center=vector(0,0,0))

# Box for orientation
device_box = box(canvas=scene1, length=1, height=0.2, width=0.5, color=color.red)

# Ball for position tracking
tracker_ball = sphere(canvas=scene2, radius=0.1, color=color.cyan, make_trail=True)

# === STATE VARIABLES ===
R = np.identity(3)  # rotation matrix
pos = np.zeros(3)   # position in world frame
vel = np.zeros(3)   # velocity in world frame

# === READ CSV ===
with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    data = list(reader)

# === MAIN LOOP ===
prev_time = None
for row in data:
    t = float(row["time"])
    ax = float(row["ax"])
    ay = float(row["ay"])
    az = float(row["az"])
    wx = float(row["wx"])
    wy = float(row["wy"])
    wz = float(row["wz"])
    
    if prev_time is None:
        prev_time = t
        continue
    
    dt = t - prev_time
    prev_time = t
    
    rate(100)  # control simulation speed
    
    # ---- ORIENTATION UPDATE ----
    omega = np.array([wx, wy, wz])  # rad/s
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-8:
        R = R @ axis_angle_to_matrix(omega, omega_mag * dt)
    
    # Update box orientation
    device_box.axis = vector(R[0,0], R[1,0], R[2,0])
    device_box.up = vector(R[0,2], R[1,2], R[2,2])
    
    # ---- POSITION UPDATE ----
    acc_device = np.array([ax, ay, az])  # m/s², in device frame
    acc_world = R @ acc_device           # rotate to world frame
    acc_world[2] -= g                    # remove gravity
    
    vel += acc_world * dt
    pos += vel * dt
    
    tracker_ball.pos = vector(*pos)
