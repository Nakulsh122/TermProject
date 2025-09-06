# imu_position.py
import serial, math, numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4
from ahrs.filters import Madgwick
import time

# ---------------- USER CONFIG ----------------
PORT = 'COM7'           # adjust
BAUD = 115200
GYRO_IN_DEG = True      # Arduino outputs deg/s
BETA = 0.25
INITIAL_SAMPLE_HZ = 100.0
MAX_POINTS = 400
ZUPT_ACCEL_THRESH = 0.12   # m/s^2 tolerance from stationary
ZUPT_GYRO_THRESH = 1.5     # deg/s gyro threshold (if below -> stationary)
HPF_ALPHA = 0.995          # high-pass smoothing factor (closer to 1 => stronger HPF)
VEL_SMOOTH = 0.9           # velocity exponential smoothing
# ---------------------------------------------

ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(1.0)

# GUI setup
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="IMU Position (Madgwick + ZUPT)")
win.resize(1200, 1000)

# 2D plots
p1 = win.addPlot(title="Acceleration (m/s²)"); p1.addLegend()
curve_ax = p1.plot(pen='r', name='ax'); curve_ay = p1.plot(pen='g', name='ay'); curve_az = p1.plot(pen='b', name='az')
win.nextRow()
p2 = win.addPlot(title="Gyroscope (deg/s)"); p2.addLegend()
curve_gx = p2.plot(pen='r', name='gx'); curve_gy = p2.plot(pen='g', name='gy'); curve_gz = p2.plot(pen='b', name='gz')
win.nextRow()
p3 = win.addPlot(title="Orientation (deg)"); p3.addLegend()
curve_roll = p3.plot(pen='r', name='roll'); curve_pitch = p3.plot(pen='g', name='pitch'); curve_yaw = p3.plot(pen='b', name='yaw')
win.nextRow()
p4 = win.addPlot(title="Position X (m)"); curve_px = p4.plot(pen='r', name='px')
win.nextRow()
p5 = win.addPlot(title="Position Y (m)"); curve_py = p5.plot(pen='g', name='py')
win.nextRow()
p6 = win.addPlot(title="XY Trajectory (top-down)"); p6.setAspectLocked(True)
curve_xy = p6.plot(pen=pg.mkPen('w', width=2))

# 3D cube window
w3d = gl.GLViewWidget(); w3d.setWindowTitle('Orientation Cube'); w3d.setCameraPosition(distance=10, azimuth=30, elevation=20)
axis = gl.GLAxisItem(); axis.setSize(3,3,3); w3d.addItem(axis)
grid = gl.GLGridItem(); grid.scale(2,2,1); w3d.addItem(grid)
verts = np.array([[1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1]])
faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]])
colors = np.array([[1,0,0,0.6]]*2 + [[0,1,0,0.6]]*2 + [[0,0,1,0.6]]*2 + [[1,1,0,0.6]]*2 + [[0,1,1,0.6]]*2 + [[1,0,1,0.6]]*2)
meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
cube = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawEdges=True, edgeColor=(1,1,1,1)); cube.scale(2,2,2); w3d.addItem(cube)
w3d.show()
win.show()

# Buffers & state
t_buf = []
ax_buf, ay_buf, az_buf = [], [], []
gx_buf, gy_buf, gz_buf = [], [], []
roll_buf, pitch_buf, yaw_buf = [], [], []
px_buf, py_buf = [], []

madgwick = Madgwick(beta=BETA, sampleperiod=1.0/INITIAL_SAMPLE_HZ)
q = np.array([1.0, 0.0, 0.0, 0.0])
last_time = None

vel = np.zeros(3)
pos = np.zeros(3)
# high-pass state
prev_a_world = np.zeros(3)
hpf_state = np.zeros(3)

def quat_to_euler_deg(q):
    w, x, y, z = q
    roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    s = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.degrees(math.asin(s))
    yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw

def cube_set_quat(q):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y), 0],
        [2*(x*y + w*z),   1-2*(x*x+z*z),   2*(y*z - w*x), 0],
        [2*(x*z - w*y),   2*(y*z + w*x),   1-2*(x*x+y*y), 0],
        [0,0,0,1]
    ], dtype=float)
    cube.resetTransform()
    cube.setTransform(QMatrix4x4(*R.flatten()))

def rotate_sensor_to_world(q, ax, ay, az):
    # R from quaternion
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ])
    return R.dot(np.array([ax, ay, az]))

def update():
    global last_time, q, vel, pos, prev_a_world, hpf_state

    # read freshest line
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
    if len(parts) != 7:
        return
    try:
        t_ms, ax, ay, az, gx, gy, gz = map(float, parts)
    except:
        return

    # time & dt
    if last_time is None:
        last_time = t_ms
        return
    dt = (t_ms - last_time) / 1000.0
    if dt <= 0 or dt > 1.0:
        last_time = t_ms
        return
    last_time = t_ms

    # update madgwick timestep
    try:
        madgwick.sampleperiod = dt
    except:
        pass

    # gyro to rad/s
    if GYRO_IN_DEG:
        g = np.radians([gx, gy, gz])
    else:
        g = np.array([gx, gy, gz], dtype=float)

    # update filter
    a_sensor = np.array([ax, ay, az])
    q = madgwick.updateIMU(q, gyr=g, acc=a_sensor)

    # orientation
    roll, pitch, yaw = quat_to_euler_deg(q)

    # world acceleration and gravity-subtraction
    a_world = rotate_sensor_to_world(q, ax, ay, az)  # in m/s^2
    a_motion = a_world - np.array([0.0, 0.0, 9.81])

    # High-pass filter to remove residual bias on a_motion
    # HPF: y[n] = alpha*(y[n-1] + x[n] - x[n-1])
    hpf_state = HPF_step = HPF_step if False else hpf_state  # placeholder (keeps linter quiet)
    # apply HPF per-axis
    global HPF_ALPHA
    for i in range(3):
        hpf_state[i] = HPF_ALPHA * (hpf_state[i] + a_motion[i] - prev_a_world[i])
    a_hpf = hpf_state.copy()
    prev_a_world = a_motion.copy()

    # Integrate velocity and position (use only X and Y)
    vel += a_hpf * dt            # integrate high-passed motion
    # smooth velocity (reduce jitter)
    vel = VEL_SMOOTH * vel + (1 - VEL_SMOOTH) * vel

    pos += vel * dt

    # ZUPT: if stationary -> zero velocity
    gyro_mag = np.linalg.norm([gx, gy, gz])
    acc_norm = np.linalg.norm(a_world)
    if (gyro_mag < ZUPT_GYRO_THRESH) and (abs(acc_norm - 9.81) < ZUPT_ACCEL_THRESH):
        vel[:] = 0.0

    # store buffers (time-based x/y plots)
    t_sec = t_ms / 1000.0
    t_buf.append(t_sec)
    ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
    gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)
    roll_buf.append(roll); pitch_buf.append(pitch); yaw_buf.append(yaw)
    px_buf.append(pos[0]); py_buf.append(pos[1])

    # keep buffers bounded
    if len(t_buf) > MAX_POINTS:
        t_buf.pop(0)
        for lst in (ax_buf, ay_buf, az_buf, gx_buf, gy_buf, gz_buf, roll_buf, pitch_buf, yaw_buf, px_buf, py_buf):
            lst.pop(0)

    # update plots
    curve_ax.setData(t_buf, ax_buf); curve_ay.setData(t_buf, ay_buf); curve_az.setData(t_buf, az_buf)
    curve_gx.setData(t_buf, gx_buf); curve_gy.setData(t_buf, gy_buf); curve_gz.setData(t_buf, gz_buf)
    curve_roll.setData(t_buf, roll_buf); curve_pitch.setData(t_buf, pitch_buf); curve_yaw.setData(t_buf, yaw_buf)
    curve_px.setData(t_buf, px_buf); curve_py.setData(t_buf, py_buf)
    curve_xy.setData(px_buf, py_buf)
    cube_set_quat(q)

# start timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(12)  # ~80 Hz GUI update

app.exec()
