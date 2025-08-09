from vpython import *
import numpy as np

# --- Scene 1: Orientation visualization ---
scene1 = canvas(title="IMU Orientation", width=600, height=400, center=vector(0,0,0), background=color.white)
box_obj = box(pos=vector(0,0,0), length=4, height=2, width=1, color=color.blue, opacity=0.6)
x_axis = arrow(pos=vector(0,0,0), axis=vector(2,0,0), color=color.red)
y_axis = arrow(pos=vector(0,0,0), axis=vector(0,2,0), color=color.green)
z_axis = arrow(pos=vector(0,0,0), axis=vector(0,0,2), color=color.blue)

# --- Scene 2: Trajectory visualization ---
scene2 = canvas(title="3D Position Tracking", width=600, height=400, center=vector(0,0,0), background=color.white)
ball_obj = sphere(pos=vector(0,0,0), radius=0.2, color=color.red, make_trail=True, trail_color=color.orange)
floor = box(pos=vector(0,-0.2,0), length=20, height=0.1, width=20, color=color.gray(0.8))

# Synthetic acceleration data function
def synthetic_accel(t):
    # Move in a circular helix path
    ax = 0.5 * np.cos(0.5 * t)
    ay = 0.5 * np.sin(0.5 * t)
    az = 0.05 * np.cos(0.2 * t)
    return vector(ax, ay, az)

# Initial velocity and position for ball
velocity = vector(0,0,0)
dt = 0.05

t = 0
while True:
    rate(50)

    # --- Scene 1: Rotate cube (orientation) ---
    roll = np.radians(20 * np.sin(t/2))
    pitch = np.radians(30 * np.sin(t/3))
    yaw = np.radians(40 * np.sin(t/4))

    rot_matrix = np.array([
        [np.cos(yaw)*np.cos(pitch),
         np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll),
         np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
        [np.sin(yaw)*np.cos(pitch),
         np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll),
         np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
        [-np.sin(pitch),
         np.cos(pitch)*np.sin(roll),
         np.cos(pitch)*np.cos(roll)]
    ])

    box_obj.axis = vector(rot_matrix[0,0], rot_matrix[1,0], rot_matrix[2,0])
    box_obj.up = vector(rot_matrix[0,1], rot_matrix[1,1], rot_matrix[2,1])

    # --- Scene 2: Position tracking ---
    accel = synthetic_accel(t)
    velocity += accel * dt
    ball_obj.pos += velocity * dt

    t += dt
