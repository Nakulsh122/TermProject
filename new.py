# imu_velocity_tracking_final_v3.py
import serial
import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4
import pyqtgraph.opengl as gl
from ahrs.filters import Madgwick
import time

# ---------------- CONFIG ----------------
PORT = 'COM7'
BAUD = 115200
GYRO_IN_DEG = True
SAMPLE_HZ = 100.0
MAX_POINTS = 2000

# Default parameters
BETA_DEFAULT = 0.25
HPF_ALPHA_DEFAULT = 0.995
LPF_ALPHA_DEFAULT = 0.9
VEL_SMOOTH_DEFAULT = 0.9
ZUPT_ACCEL_DEFAULT = 0.12
ZUPT_GYRO_DEFAULT = 1.5

# ---------------- SERIAL ----------------
ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(1.0)

# ---------------- GUI ----------------
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("IMU Velocity Tracking")
win.resize(1600, 900)
main_layout = QtWidgets.QVBoxLayout(win)

# ---------------- Sliders (Horizontal) ----------------
slider_names = ['BETA', 'HPF_ALPHA', 'LPF_ALPHA', 'VEL_SMOOTH', 'ZUPT_ACCEL', 'ZUPT_GYRO']
slider_defaults = [BETA_DEFAULT, HPF_ALPHA_DEFAULT, LPF_ALPHA_DEFAULT, VEL_SMOOTH_DEFAULT, ZUPT_ACCEL_DEFAULT, ZUPT_GYRO_DEFAULT]
sliders, labels = [], []

slider_layout = QtWidgets.QHBoxLayout()
label_layout = QtWidgets.QHBoxLayout()
main_layout.addLayout(slider_layout)
main_layout.addLayout(label_layout)

for name, default in zip(slider_names, slider_defaults):
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(999)
    slider.setValue(int(default*1000))
    slider.setFixedWidth(180)
    sliders.append(slider)
    slider_layout.addWidget(slider)
    
    label = QtWidgets.QLabel(f"{name}: {default:.3f}")
    label.setAlignment(QtCore.Qt.AlignHCenter)
    labels.append(label)
    label_layout.addWidget(label)

reset_btn = QtWidgets.QPushButton("Reset Velocity")
slider_layout.addWidget(reset_btn)

def get_slider_values():
    return [max(0.0, min(0.999, s.value()/1000.0)) for s in sliders]

def update_labels():
    for name, s, lbl in zip(slider_names, sliders, labels):
        lbl.setText(f"{name}: {s.value()/1000.0:.3f}")

for s in sliders:
    s.valueChanged.connect(update_labels)

# ---------------- Plots ----------------
plot_widget = pg.GraphicsLayoutWidget()
main_layout.addWidget(plot_widget)

p1 = plot_widget.addPlot(title="Acceleration (m/s²)"); p1.addLegend()
curve_ax = p1.plot(pen='r', name='ax'); curve_ay = p1.plot(pen='g', name='ay'); curve_az = p1.plot(pen='b', name='az')

p2 = plot_widget.addPlot(title="Gyroscope (deg/s)"); p2.addLegend()
curve_gx = p2.plot(pen='r', name='gx'); curve_gy = p2.plot(pen='g', name='gy'); curve_gz = p2.plot(pen='b', name='gz')

p3 = plot_widget.addPlot(title="Orientation (deg)"); p3.addLegend()
curve_roll = p3.plot(pen='r', name='roll'); curve_pitch = p3.plot(pen='g', name='pitch'); curve_yaw = p3.plot(pen='b', name='yaw')

plot_widget.nextRow()
p4 = plot_widget.addPlot(title="Velocity X (m/s)"); curve_vx = p4.plot(pen='r')
p5 = plot_widget.addPlot(title="Velocity Y (m/s)"); curve_vy = p5.plot(pen='g')
p6 = plot_widget.addPlot(title="XY Trajectory"); p6.setAspectLocked(True)
curve_xy = p6.plot(pen=pg.mkPen('w', width=2))

# ---------------- 3D Cube ----------------
w3d = gl.GLViewWidget()
w3d.setWindowTitle('Orientation Cube')
w3d.setCameraPosition(distance=10, azimuth=30, elevation=20)
axis = gl.GLAxisItem(); axis.setSize(3,3,3); w3d.addItem(axis)
grid = gl.GLGridItem(); grid.scale(2,2,1); w3d.addItem(grid)

verts = np.array([[1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1]])
faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]])
colors = np.array([[1,0,0,0.6]]*2 + [[0,1,0,0.6]]*2 + [[0,0,1,0.6]]*2 + [[1,1,0,0.6]]*2 + [[0,1,1,0.6]]*2 + [[1,0,1,0.6]]*2)
meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
cube = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawEdges=True, edgeColor=(1,1,1,1))
cube.scale(2,2,2)
w3d.addItem(cube)
main_layout.addWidget(w3d)

# ---------------- Buffers ----------------
t_buf, ax_buf, ay_buf, az_buf = [], [], [], []
gx_buf, gy_buf, gz_buf = [], [], []
roll_buf, pitch_buf, yaw_buf = [], [], []
vx_buf, vy_buf = [], []
pos_buf_x, pos_buf_y = [], []

madgwick = Madgwick(beta=BETA_DEFAULT, sampleperiod=1.0/SAMPLE_HZ)
q = np.array([1.0,0.0,0.0,0.0])
last_time = None
vel = np.zeros(3)
vel_prev = np.zeros(3)
pos = np.zeros(3)
prev_a_motion = np.zeros(3)
hpf_state = np.zeros(3)
a_lp_prev = np.zeros(3)

# ---------------- Functions ----------------
def quat_to_euler_deg(q):
    w,x,y,z = q
    roll = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    s = max(-1.0, min(1.0, 2*(w*y-z*x)))
    pitch = math.degrees(math.asin(s))
    yaw = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw

def cube_set_quat(q):
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 0],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 0],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), 0],
        [0,0,0,1]
    ], dtype=float)
    cube.resetTransform()
    cube.setTransform(QMatrix4x4(*R.flatten()))

def rotate_sensor_to_world(q, ax, ay, az):
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    return R.dot(np.array([ax,ay,az]))

def reset_velocity():
    global vel, vel_prev
    vel[:] = 0.0
    vel_prev[:] = 0.0
reset_btn.clicked.connect(reset_velocity)

# ---------------- Update Loop ----------------
def update():
    global last_time, q, vel, vel_prev, pos, prev_a_motion, hpf_state, a_lp_prev

    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            line = None
            break
    if not line:
        return

    parts = line.split(',')
    if len(parts)!=7:
        return
    try:
        t_ms, ax, ay, az, gx, gy, gz = map(float, parts)
    except:
        return

    if last_time is None:
        last_time = t_ms
        return
    dt = (t_ms-last_time)/1000.0
    if dt <= 0 or dt > 1.0:
        last_time = t_ms
        return
    last_time = t_ms

    beta, HPF_ALPHA, LPF_ALPHA, VEL_SMOOTH, ZUPT_ACCEL, ZUPT_GYRO = get_slider_values()
    madgwick.beta = beta

    g = np.radians([gx,gy,gz]) if GYRO_IN_DEG else np.array([gx,gy,gz])
    q_new = madgwick.updateIMU(q, gyr=g, acc=np.array([ax,ay,az]))
    if q_new is not None:
        q = q_new
    roll, pitch, yaw = quat_to_euler_deg(q)

    a_world = rotate_sensor_to_world(q, ax, ay, az)
    a_motion = a_world - np.array([0,0,9.81])

    a_lp = LPF_ALPHA*a_lp_prev + (1-LPF_ALPHA)*a_motion
    a_lp_prev[:] = a_lp
    hpf_state[:] = HPF_ALPHA*(hpf_state + a_lp - prev_a_motion)
    a_hpf = hpf_state.copy()
    prev_a_motion[:] = a_lp

    # ---------------- Velocity ----------------
    vel_new = VEL_SMOOTH*vel_prev + (1-VEL_SMOOTH)*(vel + a_hpf*dt)

    # ZUPT detection
    zupt = np.linalg.norm([gx,gy,gz])<ZUPT_GYRO and abs(np.linalg.norm(a_world)-9.81)<ZUPT_ACCEL
    if zupt:
        vel[:] = 0.0
        vel_prev[:] = 0.0
    else:
        vel[:] = vel_new
        vel_prev[:] = vel_new

    # Integrate position continuously
    pos[:2] += vel_new[:2]*dt

    # ---------------- Buffers ----------------
    t_sec = t_ms/1000.0
    t_buf.append(t_sec)
    ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
    gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)
    roll_buf.append(roll); pitch_buf.append(pitch); yaw_buf.append(yaw)
    vx_buf.append(vel_new[0]); vy_buf.append(vel_new[1])
    pos_buf_x.append(pos[0]); pos_buf_y.append(pos[1])

    if len(t_buf) > MAX_POINTS:
        for buf in (t_buf, ax_buf, ay_buf, az_buf, gx_buf, gy_buf, gz_buf, roll_buf, pitch_buf, yaw_buf, vx_buf, vy_buf, pos_buf_x, pos_buf_y):
            buf.pop(0)

    # ---------------- Last 1s view ----------------
    t_min = t_sec - 1.0
    idx_start = next((i for i,v in enumerate(t_buf) if v>=t_min),0)
    t_view = t_buf[idx_start:]
    ax_view, ay_view, az_view = ax_buf[idx_start:], ay_buf[idx_start:], az_buf[idx_start:]
    gx_view, gy_view, gz_view = gx_buf[idx_start:], gy_buf[idx_start:], gz_buf[idx_start:]
    roll_view, pitch_view, yaw_view = roll_buf[idx_start:], pitch_buf[idx_start:], yaw_buf[idx_start:]
    vx_view, vy_view = vx_buf[idx_start:], vy_buf[idx_start:]
    pos_x_view, pos_y_view = pos_buf_x[idx_start:], pos_buf_y[idx_start:]

    curve_ax.setData(t_view, ax_view); curve_ay.setData(t_view, ay_view); curve_az.setData(t_view, az_view)
    curve_gx.setData(t_view, gx_view); curve_gy.setData(t_view, gy_view); curve_gz.setData(t_view, gz_view)
    curve_roll.setData(t_view, roll_view); curve_pitch.setData(t_view, pitch_view); curve_yaw.setData(t_view, yaw_view)
    curve_vx.setData(t_view, vx_view); curve_vy.setData(t_view, vy_view)
    curve_xy.setData(pos_x_view, pos_y_view)

    for p in (p1,p2,p3,p4,p5):
        p.setXRange(max(t_min,t_view[0]), t_sec, padding=0)

    cube_set_quat(q)

# ---------------- Timer ----------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(12)

win.show()
app.exec()
