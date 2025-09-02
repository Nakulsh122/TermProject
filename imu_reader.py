import serial, math, numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4

# === Serial Setup ===
ser = serial.Serial('COM7', 115200, timeout=0)

# Complementary filter params
alpha = 0.98
roll, pitch, yaw = 0.0, 0.0, 0.0
last_time = None

# Buffers
MAX_POINTS = 300
t_data, ax_data, ay_data, az_data = [], [], [], []
roll_data, pitch_data, yaw_data = [], [], []

# === Create QApplication ===
app = QtWidgets.QApplication([])

# === Window 1: Graphs ===
win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Realtime Data")
win.resize(1000,600)

# Plot 1: Accel
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

# === Window 2: 3D Cube ===
w3d = gl.GLViewWidget()
w3d.setWindowTitle('MPU6050 Orientation Cube')
w3d.setCameraPosition(distance=10, azimuth=45, elevation=20)
w3d.show()

# Add axis + grid
axis = gl.GLAxisItem()
axis.setSize(3,3,3)
w3d.addItem(axis)

grid = gl.GLGridItem()
grid.scale(2,2,1)
w3d.addItem(grid)

# Cube mesh
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
cube.scale(2,2,2)   # make it visible
w3d.addItem(cube)

# === Complementary Filter ===
def complementary_filter(ax, ay, az, gx, gy, gz, dt, alpha):
    global roll, pitch, yaw
    ax_g, ay_g, az_g = ax/9.81, ay/9.81, az/9.81

    acc_roll = math.degrees(math.atan2(ay_g, az_g))
    acc_pitch = math.degrees(math.atan2(-ax_g, math.sqrt(ay_g*ay_g + az_g*az_g)))

    roll += gx * dt
    pitch += gy * dt
    yaw += gz * dt

    roll = alpha * roll + (1-alpha)*acc_roll
    pitch = alpha * pitch + (1-alpha)*acc_pitch

    return roll, pitch, yaw

# === Update Loop ===
def update():
    global last_time, roll, pitch, yaw
    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            pass

    if line:
        parts = line.split(",")
        if len(parts) == 7:
            try:
                t, ax, ay, az, gx, gy, gz = map(float, parts)

                if last_time is None:
                    last_time = t
                    return

                dt = (t - last_time)/1000.0
                last_time = t

                roll, pitch, yaw = complementary_filter(ax, ay, az, gx, gy, gz, dt, alpha)

                # store data
                t_data.append(t/1000.0)
                ax_data.append(ax); ay_data.append(ay); az_data.append(az)
                roll_data.append(roll); pitch_data.append(pitch); yaw_data.append(yaw)

                if len(t_data) > MAX_POINTS:
                    t_data.pop(0); ax_data.pop(0); ay_data.pop(0); az_data.pop(0)
                    roll_data.pop(0); pitch_data.pop(0); yaw_data.pop(0)

                # update plots
                curve_ax.setData(t_data, ax_data)
                curve_ay.setData(t_data, ay_data)
                curve_az.setData(t_data, az_data)
                curve_roll.setData(t_data, roll_data)
                curve_pitch.setData(t_data, pitch_data)
                curve_yaw.setData(t_data, yaw_data)

                # update cube
                cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
                cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
                cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

                R = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, 0],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0],
                    [-sp,   cp*sr,            cp*cr,            0],
                    [0,0,0,1]
                ])

                cube.resetTransform()
                cube.setTransform(QMatrix4x4(*R.flatten()))

            except ValueError:
                pass

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

app.exec()
