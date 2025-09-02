import serial, math, numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4
from ahrs.filters import Madgwick

# ========= USER SETTINGS =========
PORT = 'COM7'
BAUD = 115200

# Set this based on what your ESP sends for gx,gy,gz
# If your Arduino divides by 131 (±250 dps) and prints “deg/s”, set True.
# If you already send rad/s from the ESP, set False (so we won't convert again).
GYRO_IN_DEG = True

# Madgwick parameters
BETA = 0.25       # try 0.2–0.6; higher = more accel influence, snappier but noisier
INITIAL_SAMPLE_HZ = 100.0  # just a starting guess; we’ll update every frame from timestamps
# =================================

ser = serial.Serial(PORT, BAUD, timeout=0)

# Create app
app = QtWidgets.QApplication([])

# ----- Window 1: Graphs -----
win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Realtime (Madgwick)")
win.resize(1000, 600)

# Plot 1: Acceleration
p1 = win.addPlot(title="Acceleration (m/s²)")
p1.addLegend()
curve_ax = p1.plot(pen='r', name="ax")
curve_ay = p1.plot(pen='g', name="ay")
curve_az = p1.plot(pen='b', name="az")

win.nextRow()

# Plot 2: Orientation
p2 = win.addPlot(title="Orientation (deg)")
p2.addLegend()
curve_roll = p2.plot(pen='r', name="Roll")
curve_pitch = p2.plot(pen='g', name="Pitch")
curve_yaw = p2.plot(pen='b', name="Yaw")

# ----- Window 2: 3D Cube -----
w3d = gl.GLViewWidget()
w3d.setWindowTitle('Orientation Cube (Madgwick)')
w3d.setCameraPosition(distance=10, azimuth=45, elevation=20)
w3d.show()

axis = gl.GLAxisItem()
axis.setSize(3,3,3)
w3d.addItem(axis)

grid = gl.GLGridItem()
grid.scale(2,2,1)
w3d.addItem(grid)

# Cube
verts = np.array([
    [1,1,1], [1,1,-1], [1,-1,-1], [1,-1,1],
    [-1,1,1], [-1,1,-1], [-1,-1,-1], [-1,-1,1]
])
faces = np.array([
    [0,1,2], [0,2,3],
    [4,5,6], [4,6,7],
    [0,1,5], [0,5,4],
    [2,3,7], [2,7,6],
    [1,2,6], [1,6,5],
    [0,3,7], [0,7,4]
])
colors = np.array([[1,0,0,0.6]]*2 + [[0,1,0,0.6]]*2 +
                  [[0,0,1,0.6]]*2 + [[1,1,0,0.6]]*2 +
                  [[0,1,1,0.6]]*2 + [[1,0,1,0.6]]*2)
meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
cube = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawEdges=True, edgeColor=(1,1,1,1))
cube.scale(2,2,2)
w3d.addItem(cube)

# Buffers
MAX_POINTS = 300
t_data, ax_data, ay_data, az_data = [], [], [], []
roll_data, pitch_data, yaw_data = [], [], []


# Madgwick filter init
madgwick = Madgwick(beta=BETA, sampleperiod=1.0/INITIAL_SAMPLE_HZ)
q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

last_time = None

def quat_to_euler_deg(q):
    w, x, y, z = q
    # roll (x)
    roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    # pitch (y) – clamp for asin domain
    s = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.degrees(math.asin(s))
    # yaw (z)
    yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw

def cube_set_quat(q):
    # Quaternion to rotation matrix
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y), 0],
        [2*(x*y + w*z),   1-2*(x*x+z*z),   2*(y*z - w*x), 0],
        [2*(x*z - w*y),   2*(y*z + w*x),   1-2*(x*x+y*y), 0],
        [0,0,0,1]
    ], dtype=float)
    cube.resetTransform()
    cube.setTransform(QMatrix4x4(*R.flatten()))

def update():
    global last_time, q, madgwick

    # read the most recent full line available
    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            line = None
            break

    if not line:
        return

    parts = line.split(",")
    if len(parts) != 7:
        return

    try:
        t_ms, ax, ay, az, gx, gy, gz = map(float, parts)
    except ValueError:
        return

    if last_time is None:
        last_time = t_ms
        return

    dt = (t_ms - last_time) / 1000.0
    if dt <= 0:
        return
    last_time = t_ms

    # Keep Madgwick's internal sample period updated
    # (Different versions use 'sampleperiod' or 'dt'; this covers both.)
    try:
        madgwick.sampleperiod = dt
    except Exception:
        pass
    try:
        madgwick.dt = dt
    except Exception:
        pass

    # Prepare sensor vectors
    # Madgwick expects gyro in rad/s
    if GYRO_IN_DEG:
        g = np.radians([gx, gy, gz])
    else:
        g = np.array([gx, gy, gz], dtype=float)

    # Accel can be in m/s^2; filter normalizes internally
    a = np.array([ax, ay, az], dtype=float)

    # Update filter
    q = madgwick.updateIMU(q, gyr=g, acc=a)

    # Euler for plots
    roll, pitch, yaw = quat_to_euler_deg(q)

    # Store and keep last N points
    t_data.append(t_ms/1000.0)
    ax_data.append(ax); ay_data.append(ay); az_data.append(az)
    roll_data.append(roll); pitch_data.append(pitch); yaw_data.append(yaw)
    if len(t_data) > MAX_POINTS:
        t_data.pop(0); ax_data.pop(0); ay_data.pop(0); az_data.pop(0)
        roll_data.pop(0); pitch_data.pop(0); yaw_data.pop(0)

    # Update plots
    curve_ax.setData(t_data, ax_data)
    curve_ay.setData(t_data, ay_data)
    curve_az.setData(t_data, az_data)
    curve_roll.setData(t_data, roll_data)
    curve_pitch.setData(t_data, pitch_data)
    curve_yaw.setData(t_data, yaw_data)

    # Update cube
    cube_set_quat(q)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(15)  # ~66 Hz UI updates

app.exec()
