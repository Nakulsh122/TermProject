import serial, math, numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4
from ahrs.filters import Madgwick
import time

# ========= USER SETTINGS =========
PORT = 'COM7'
BAUD = 115200
GYRO_IN_DEG = True
BETA = 0.25
INITIAL_SAMPLE_HZ = 100.0
CALIB_SAMPLES = 500
MAX_POINTS = 300
# =================================

# ------------------- Serial -------------------
ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(2)

# ------------------- Calibration -------------------
print("Calibrating... keep sensor still")
accel_offset = np.zeros(3)
gyro_offset = np.zeros(3)
count = 0
while count < CALIB_SAMPLES:
    line = ser.readline().decode(errors='ignore').strip()
    if not line or ',' not in line:
        continue
    parts = line.split(',')
    if len(parts) != 7:
        continue
    _, ax, ay, az, gx, gy, gz = map(float, parts)
    accel_offset += np.array([ax, ay, az])
    gyro_offset += np.array([gx, gy, gz])
    count += 1

accel_offset /= CALIB_SAMPLES
gyro_offset /= CALIB_SAMPLES
accel_offset[2] -= 9.81  # gravity
print("Calibration done")

# ------------------- Madgwick -------------------
madgwick = Madgwick(beta=BETA, sampleperiod=1.0/INITIAL_SAMPLE_HZ)
q = np.array([1.0, 0.0, 0.0, 0.0])
last_time = None

# ------------------- Buffers -------------------
t_data, ax_data, ay_data, az_data = [], [], [], []
gx_data, gy_data, gz_data = [], [], []
roll_data, pitch_data, yaw_data = [], [], []

# ------------------- GUI -------------------
app = QtWidgets.QApplication([])

# 2D Plots
win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Realtime (Madgwick)")
win.resize(1000, 900)

# Acceleration
p1 = win.addPlot(title="Acceleration (m/s²)")
p1.addLegend()
curve_ax = p1.plot(pen='r', name="ax")
curve_ay = p1.plot(pen='g', name="ay")
curve_az = p1.plot(pen='b', name="az")
win.nextRow()

# Gyroscope
p2 = win.addPlot(title="Gyroscope (deg/s)")
p2.addLegend()
curve_gx = p2.plot(pen='r', name="gx")
curve_gy = p2.plot(pen='g', name="gy")
curve_gz = p2.plot(pen='b', name="gz")
win.nextRow()

# Orientation
p3 = win.addPlot(title="Orientation (deg)")
p3.addLegend()
curve_roll = p3.plot(pen='r', name="Roll")
curve_pitch = p3.plot(pen='g', name="Pitch")
curve_yaw = p3.plot(pen='b', name="Yaw")

# 3D Cube
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

verts = np.array([
    [1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],
    [-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1]
])
faces = np.array([
    [0,1,2],[0,2,3],[4,5,6],[4,6,7],
    [0,1,5],[0,5,4],[2,3,7],[2,7,6],
    [1,2,6],[1,6,5],[0,3,7],[0,7,4]
])
colors = np.array([[1,0,0,0.6]]*2 + [[0,1,0,0.6]]*2 +
                  [[0,0,1,0.6]]*2 + [[1,1,0,0.6]]*2 +
                  [[0,1,1,0.6]]*2 + [[1,0,1,0.6]]*2)
meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
cube = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawEdges=True, edgeColor=(1,1,1,1))
cube.scale(2,2,2)
w3d.addItem(cube)

# ------------------- Helper Functions -------------------
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
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x), 0],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y), 0],
        [0,0,0,1]
    ], dtype=float)
    cube.resetTransform()
    cube.setTransform(QMatrix4x4(*R.flatten()))

# ------------------- Update Loop -------------------
def update():
    global last_time, q, madgwick
    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
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

    # Apply calibration
    ax -= accel_offset[0]; ay -= accel_offset[1]; az -= accel_offset[2]
    gx -= gyro_offset[0]; gy -= gyro_offset[1]; gz -= gyro_offset[2]

    # Time delta
    global last_time
    if last_time is None:
        last_time = t_ms
        return
    dt = (t_ms - last_time)/1000.0
    if dt <= 0: return
    last_time = t_ms
    madgwick.sampleperiod = dt

    # Gyro rad/s
    g = np.radians([gx, gy, gz]) if GYRO_IN_DEG else np.array([gx,gy,gz])
    a = np.array([ax, ay, az])
    q = madgwick.updateIMU(q, gyr=g, acc=a)
    roll, pitch, yaw = quat_to_euler_deg(q)

    # Store data
    t_sec = t_ms/1000.0
    t_data.append(t_sec); ax_data.append(ax); ay_data.append(ay); az_data.append(az)
    gx_data.append(gx); gy_data.append(gy); gz_data.append(gz)
    roll_data.append(roll); pitch_data.append(pitch); yaw_data.append(yaw)
    if len(t_data) > MAX_POINTS:
        t_data.pop(0); ax_data.pop(0); ay_data.pop(0); az_data.pop(0)
        gx_data.pop(0); gy_data.pop(0); gz_data.pop(0)
        roll_data.pop(0); pitch_data.pop(0); yaw_data.pop(0)

    # Update plots
    curve_ax.setData(t_data, ax_data)
    curve_ay.setData(t_data, ay_data)
    curve_az.setData(t_data, az_data)
    curve_gx.setData(t_data, gx_data)
    curve_gy.setData(t_data, gy_data)
    curve_gz.setData(t_data, gz_data)
    curve_roll.setData(t_data, roll_data)
    curve_pitch.setData(t_data, pitch_data)
    curve_yaw.setData(t_data, yaw_data)

    # Update cube
    cube_set_quat(q)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(15)  # ~66Hz

app.exec()
