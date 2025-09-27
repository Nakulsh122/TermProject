# gesture_tracking_imu.py
import serial
import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from ahrs.filters import Madgwick
import time
from collections import deque
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
PORT = 'COM7'
BAUD = 115200
GYRO_IN_DEG = True
SAMPLE_HZ = 100.0
GESTURE_WINDOW = 2.0  # seconds
MAX_GESTURE_POINTS = int(GESTURE_WINDOW * SAMPLE_HZ)

# Gesture-specific parameters
BETA_DEFAULT = 0.1  # Lower for smoother orientation during gestures
HPF_ALPHA_DEFAULT = 0.96  # Less aggressive for gesture motion
LPF_ALPHA_DEFAULT = 0.7   # More responsive for gesture dynamics
MOTION_THRESHOLD = 0.5    # m/s² threshold to detect gesture start
STILLNESS_THRESHOLD = 0.15  # m/s² threshold for gesture end
GESTURE_MIN_DURATION = 0.3  # Minimum gesture duration (seconds)
GESTURE_MAX_DURATION = 2.5  # Maximum gesture duration (seconds)

# Additional parameters for gesture analysis
VEL_DECAY = 0.98  # Velocity decay for gesture tracking
POS_SMOOTHING = 0.85  # Position smoothing for gestures

# ---------------- SERIAL ----------------
ser = serial.Serial(PORT, BAUD, timeout=0)
time.sleep(1.0)

# ---------------- GUI ----------------
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("Gesture Tracking with IMU")
win.resize(1800, 1100)
main_layout = QtWidgets.QVBoxLayout(win)

# ---------------- Gesture Controls ----------------
control_layout = QtWidgets.QHBoxLayout()
main_layout.addLayout(control_layout)

# Gesture detection controls
gesture_group = QtWidgets.QGroupBox("Gesture Detection")
gesture_layout = QtWidgets.QHBoxLayout(gesture_group)

start_gesture_btn = QtWidgets.QPushButton("Start Recording")
stop_gesture_btn = QtWidgets.QPushButton("Stop Recording")
clear_gesture_btn = QtWidgets.QPushButton("Clear Gesture")
save_gesture_btn = QtWidgets.QPushButton("Save Gesture")
auto_detect_cb = QtWidgets.QCheckBox("Auto Detect")
auto_detect_cb.setChecked(True)

gesture_layout.addWidget(start_gesture_btn)
gesture_layout.addWidget(stop_gesture_btn)
gesture_layout.addWidget(clear_gesture_btn)
gesture_layout.addWidget(save_gesture_btn)
gesture_layout.addWidget(auto_detect_cb)

control_layout.addWidget(gesture_group)

# Parameter controls
param_group = QtWidgets.QGroupBox("Parameters")
param_layout = QtWidgets.QGridLayout(param_group)

slider_names = ['BETA', 'HPF_ALPHA', 'LPF_ALPHA', 'VEL_DECAY', 'MOTION_THRESH', 'STILLNESS_THRESH']
slider_defaults = [BETA_DEFAULT, HPF_ALPHA_DEFAULT, LPF_ALPHA_DEFAULT, VEL_DECAY, MOTION_THRESHOLD, STILLNESS_THRESHOLD]
slider_ranges = [(0.01, 0.5), (0.9, 0.999), (0.1, 0.95), (0.9, 0.999), (0.1, 2.0), (0.05, 0.5)]
sliders, labels = [], []

def scale_slider_value(value, min_val, max_val):
    return min_val + (max_val - min_val) * (value / 999.0)

def inverse_scale_slider_value(param_val, min_val, max_val):
    return int(999 * (param_val - min_val) / (max_val - min_val))

for i, (name, default, (min_val, max_val)) in enumerate(zip(slider_names, slider_defaults, slider_ranges)):
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(999)
    slider.setValue(inverse_scale_slider_value(default, min_val, max_val))
    slider.setFixedWidth(120)
    sliders.append(slider)
    
    label = QtWidgets.QLabel(f"{name}: {default:.3f}")
    labels.append(label)
    
    param_layout.addWidget(QtWidgets.QLabel(name), i//3, (i%3)*2)
    param_layout.addWidget(slider, i//3, (i%3)*2 + 1)
    param_layout.addWidget(label, (i//3) + 2, (i%3)*2, 1, 2)

control_layout.addWidget(param_group)

def get_slider_values():
    return [scale_slider_value(s.value(), min_val, max_val) 
            for s, (min_val, max_val) in zip(sliders, slider_ranges)]

def update_labels():
    for name, s, lbl, (min_val, max_val) in zip(slider_names, sliders, labels, slider_ranges):
        val = scale_slider_value(s.value(), min_val, max_val)
        lbl.setText(f"{name}: {val:.3f}")

for s in sliders:
    s.valueChanged.connect(update_labels)

# Status and metrics
status_layout = QtWidgets.QHBoxLayout()
status_label = QtWidgets.QLabel("Status: Ready for gesture")
gesture_info_label = QtWidgets.QLabel("Gesture: None")
metrics_label = QtWidgets.QLabel("Metrics: -")
status_layout.addWidget(status_label)
status_layout.addWidget(gesture_info_label)
status_layout.addWidget(metrics_label)
main_layout.addLayout(status_layout)

# ---------------- Enhanced Plots for Gesture Analysis ----------------
plot_widget = pg.GraphicsLayoutWidget()
main_layout.addWidget(plot_widget)

# Top row - Raw sensor data with gesture highlighting
p1 = plot_widget.addPlot(title="Acceleration (m/s²)"); p1.addLegend()
curve_ax = p1.plot(pen='r', name='ax'); curve_ay = p1.plot(pen='g', name='ay'); curve_az = p1.plot(pen='b', name='az')
gesture_region_acc = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(255, 255, 0, 50))
p1.addItem(gesture_region_acc)
p1.setYRange(-20, 20)

p2 = plot_widget.addPlot(title="Angular Velocity (deg/s)"); p2.addLegend()
curve_gx = p2.plot(pen='r', name='gx'); curve_gy = p2.plot(pen='g', name='gy'); curve_gz = p2.plot(pen='b', name='gz')
gesture_region_gyro = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(255, 255, 0, 50))
p2.addItem(gesture_region_gyro)

p3 = plot_widget.addPlot(title="Motion Magnitude"); 
curve_motion_mag = p3.plot(pen='w', width=2)
motion_threshold_line = pg.InfiniteLine(pos=MOTION_THRESHOLD, angle=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
stillness_threshold_line = pg.InfiniteLine(pos=STILLNESS_THRESHOLD, angle=0, pen=pg.mkPen('g', style=QtCore.Qt.DashLine))
p3.addItem(motion_threshold_line)
p3.addItem(stillness_threshold_line)

# Middle row - Gesture-specific data
plot_widget.nextRow()
p4 = plot_widget.addPlot(title="Gesture Velocity (m/s)"); p4.addLegend()
curve_vx = p4.plot(pen='r', name='vx'); curve_vy = p4.plot(pen='g', name='vy'); curve_vz = p4.plot(pen='b', name='vz')
p4.setYRange(-3, 3)

p5 = plot_widget.addPlot(title="Gesture Orientation (deg)"); p5.addLegend()
curve_roll = p5.plot(pen='r', name='roll'); curve_pitch = p5.plot(pen='g', name='pitch'); curve_yaw = p5.plot(pen='b', name='yaw')

p6 = plot_widget.addPlot(title="Gesture State")
curve_gesture_state = p6.plot(pen='y', width=3)
p6.setYRange(-0.1, 2.1)

# Bottom row - 3D trajectory and analysis
plot_widget.nextRow()
p7 = plot_widget.addPlot(title="XY Gesture Trajectory")
p7.setAspectLocked(True)
curve_xy_current = p7.plot(pen=pg.mkPen('w', width=2))
curve_xy_gesture = p7.plot(pen=pg.mkPen('y', width=3))
scatter_start = pg.ScatterPlotItem(pos=[(0,0)], pen='g', brush='g', size=12, symbol='o')
scatter_end = pg.ScatterPlotItem(pos=[(0,0)], pen='r', brush='r', size=12, symbol='s')
p7.addItem(scatter_start)
p7.addItem(scatter_end)

p8 = plot_widget.addPlot(title="XZ Gesture Trajectory")
p8.setAspectLocked(True)
curve_xz_current = p8.plot(pen=pg.mkPen('w', width=2))
curve_xz_gesture = p8.plot(pen=pg.mkPen('y', width=3))

p9 = plot_widget.addPlot(title="Gesture Features")
curve_speed_profile = p9.plot(pen='c', name='Speed')
curve_curvature = p9.plot(pen='m', name='Curvature')

# ---------------- Gesture Tracking Data Structures ----------------
# Real-time buffers (circular for efficiency)
buffer_size = int(5.0 * SAMPLE_HZ)  # 5 second buffer
t_buf = deque(maxlen=buffer_size)
ax_buf = deque(maxlen=buffer_size); ay_buf = deque(maxlen=buffer_size); az_buf = deque(maxlen=buffer_size)
gx_buf = deque(maxlen=buffer_size); gy_buf = deque(maxlen=buffer_size); gz_buf = deque(maxlen=buffer_size)
roll_buf = deque(maxlen=buffer_size); pitch_buf = deque(maxlen=buffer_size); yaw_buf = deque(maxlen=buffer_size)
motion_mag_buf = deque(maxlen=buffer_size)
gesture_state_buf = deque(maxlen=buffer_size)

# Gesture-specific buffers
vx_buf = deque(maxlen=buffer_size); vy_buf = deque(maxlen=buffer_size); vz_buf = deque(maxlen=buffer_size)
pos_x_buf = deque(maxlen=buffer_size); pos_y_buf = deque(maxlen=buffer_size); pos_z_buf = deque(maxlen=buffer_size)

# Gesture recording
gesture_recording = False
gesture_data = []
gesture_start_time = None
gesture_end_time = None
current_gesture = {
    'timestamps': [],
    'positions': [],
    'velocities': [],
    'orientations': [],
    'accelerations': [],
    'angular_velocities': []
}

# Filter states
madgwick = Madgwick(beta=BETA_DEFAULT, sampleperiod=1.0/SAMPLE_HZ)
q = np.array([1.0, 0.0, 0.0, 0.0])
last_time = None

# Motion tracking
velocity = np.zeros(3)
position = np.zeros(3)
prev_accel_world = np.zeros(3)

# Gesture detection state
gesture_state = 0  # 0: idle, 1: detecting, 2: recording
motion_start_time = None
last_significant_motion = None

# Calibration
accel_bias = np.zeros(3)
gyro_bias = np.zeros(3)

# Gesture metrics
gesture_metrics = {
    'duration': 0,
    'path_length': 0,
    'max_speed': 0,
    'avg_speed': 0,
    'displacement': 0
}

# ---------------- Gesture Functions ----------------
def calculate_motion_magnitude(ax, ay, az, gx, gy, gz):
    """Calculate combined motion magnitude for gesture detection"""
    accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)
    # Weight acceleration more heavily for gesture detection
    return accel_mag + 0.3 * gyro_mag

def detect_gesture_boundaries(motion_magnitudes, timestamps, motion_thresh, stillness_thresh):
    """Auto-detect gesture start and end based on motion"""
    if len(motion_magnitudes) < 10:
        return None, None
    
    # Find motion peaks
    above_threshold = np.array(motion_magnitudes) > motion_thresh
    
    if not np.any(above_threshold):
        return None, None
    
    # Find first significant motion
    start_idx = np.where(above_threshold)[0][0]
    
    # Find last significant motion (looking backwards from recent data)
    end_candidates = np.where(np.array(motion_magnitudes[-50:]) < stillness_thresh)[0]
    if len(end_candidates) > 5:  # Need sustained stillness
        end_idx = len(motion_magnitudes) - 50 + end_candidates[0]
    else:
        end_idx = len(motion_magnitudes) - 1
    
    if end_idx <= start_idx or (end_idx - start_idx) < 10:
        return None, None
    
    return timestamps[start_idx] if start_idx < len(timestamps) else None, \
           timestamps[end_idx] if end_idx < len(timestamps) else None

def calculate_gesture_metrics(positions, velocities, timestamps):
    """Calculate gesture analysis metrics"""
    if len(positions) < 2:
        return gesture_metrics
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Duration
    duration = timestamps[-1] - timestamps[0]
    
    # Path length
    path_length = 0
    for i in range(1, len(positions)):
        path_length += np.linalg.norm(positions[i] - positions[i-1])
    
    # Speed statistics
    speeds = np.linalg.norm(velocities, axis=1)
    max_speed = np.max(speeds)
    avg_speed = np.mean(speeds)
    
    # Total displacement
    displacement = np.linalg.norm(positions[-1] - positions[0])
    
    return {
        'duration': duration,
        'path_length': path_length,
        'max_speed': max_speed,
        'avg_speed': avg_speed,
        'displacement': displacement
    }

def reset_gesture():
    """Reset current gesture data"""
    global current_gesture, position, velocity, gesture_metrics
    current_gesture = {
        'timestamps': [],
        'positions': [],
        'velocities': [],
        'orientations': [],
        'accelerations': [],
        'angular_velocities': []
    }
    position[:] = 0
    velocity[:] = 0
    gesture_metrics = {'duration': 0, 'path_length': 0, 'max_speed': 0, 'avg_speed': 0, 'displacement': 0}

def save_gesture_data():
    """Save current gesture to file"""
    if not current_gesture['timestamps']:
        status_label.setText("No gesture data to save")
        return
    
    timestamp = int(time.time())
    filename = f"gesture_{timestamp}.csv"
    
    try:
        with open(filename, 'w') as f:
            f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,roll,pitch,yaw,ax,ay,az,gx,gy,gz\n")
            for i in range(len(current_gesture['timestamps'])):
                row = [
                    current_gesture['timestamps'][i],
                    current_gesture['positions'][i][0], current_gesture['positions'][i][1], current_gesture['positions'][i][2],
                    current_gesture['velocities'][i][0], current_gesture['velocities'][i][1], current_gesture['velocities'][i][2],
                    current_gesture['orientations'][i][0], current_gesture['orientations'][i][1], current_gesture['orientations'][i][2],
                    current_gesture['accelerations'][i][0], current_gesture['accelerations'][i][1], current_gesture['accelerations'][i][2],
                    current_gesture['angular_velocities'][i][0], current_gesture['angular_velocities'][i][1], current_gesture['angular_velocities'][i][2]
                ]
                f.write(','.join(map(str, row)) + '\n')
        
        status_label.setText(f"Gesture saved to {filename}")
        
        # Also save metrics
        with open(f"gesture_metrics_{timestamp}.txt", 'w') as f:
            for key, value in gesture_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
                
    except Exception as e:
        status_label.setText(f"Error saving: {str(e)}")

def start_manual_recording():
    """Start manual gesture recording"""
    global gesture_recording, gesture_start_time
    gesture_recording = True
    gesture_start_time = time.time()
    reset_gesture()
    status_label.setText("Recording gesture manually...")

def stop_manual_recording():
    """Stop manual gesture recording"""
    global gesture_recording, gesture_end_time
    gesture_recording = False
    gesture_end_time = time.time()
    if current_gesture['timestamps']:
        global gesture_metrics
        gesture_metrics = calculate_gesture_metrics(
            current_gesture['positions'],
            current_gesture['velocities'], 
            current_gesture['timestamps']
        )
    status_label.setText("Manual recording stopped")

# Connect buttons
start_gesture_btn.clicked.connect(start_manual_recording)
stop_gesture_btn.clicked.connect(stop_manual_recording)
clear_gesture_btn.clicked.connect(reset_gesture)
save_gesture_btn.clicked.connect(save_gesture_data)

# ---------------- Enhanced Update Loop for Gestures ----------------
def update():
    global last_time, q, velocity, position, prev_accel_world
    global gesture_recording, gesture_state, motion_start_time, last_significant_motion
    global accel_bias, gyro_bias, gesture_metrics

    # Read serial data
    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            line = None
            break
    
    if not line:
        return

    # Parse data
    parts = line.split(',')
    if len(parts) != 7:
        return
    
    try:
        t_ms, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw = map(float, parts)
    except:
        return

    # Time management
    if last_time is None:
        last_time = t_ms
        return
    
    dt = (t_ms - last_time) / 1000.0
    if dt <= 0 or dt > 0.1:
        last_time = t_ms
        return
    last_time = t_ms
    t_sec = t_ms / 1000.0

    # Get parameters
    beta, HPF_ALPHA, LPF_ALPHA, VEL_DECAY, MOTION_THRESH, STILLNESS_THRESH = get_slider_values()
    madgwick.beta = beta

    # Apply simple bias correction (could be calibrated)
    ax = ax_raw - accel_bias[0]
    ay = ay_raw - accel_bias[1] 
    az = az_raw - accel_bias[2] 
    gx = gx_raw - gyro_bias[0]
    gy = gy_raw - gyro_bias[1]
    gz = gz_raw - gyro_bias[2]

    # Update orientation
    g_rad = np.radians([gx, gy, gz]) if GYRO_IN_DEG else np.array([gx, gy, gz])
    q_new = madgwick.updateIMU(q, gyr=g_rad, acc=np.array([ax, ay, az]))
    if q_new is not None:
        q = q_new

    # Calculate Euler angles
    w, x, y, z = q
    roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    sinp = 2*(w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))
    yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

    # Transform to world coordinates
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    accel_world = R @ np.array([ax, ay, az])
    accel_world[2] -= 9.81  # Remove gravity

    # Gesture-optimized integration with decay
    velocity = velocity * VEL_DECAY + accel_world * dt
    position = position + velocity * dt

    # Calculate motion magnitude for gesture detection
    motion_mag = calculate_motion_magnitude(ax, ay, az, gx, gy, gz)

    # Auto gesture detection
    if auto_detect_cb.isChecked():
        if gesture_state == 0:  # Idle
            if motion_mag > MOTION_THRESH:
                gesture_state = 1
                motion_start_time = t_sec
                reset_gesture()
                status_label.setText("Motion detected - starting gesture capture...")
        
        elif gesture_state == 1:  # Detecting
            if motion_mag > MOTION_THRESH:
                last_significant_motion = t_sec
            
            # Check if we should start recording
            if last_significant_motion and (t_sec - motion_start_time) > 0.1:
                gesture_state = 2
                gesture_recording = True
                status_label.setText("Recording gesture automatically...")
            
            # Timeout if no significant motion
            if (t_sec - motion_start_time) > 1.0:
                gesture_state = 0
                status_label.setText("No gesture detected - returning to idle")
        
        elif gesture_state == 2:  # Recording
            if last_significant_motion and (t_sec - last_significant_motion) > 0.5:
                # End of gesture
                gesture_recording = False
                gesture_state = 0
                if current_gesture['timestamps']:
                    gesture_metrics = calculate_gesture_metrics(
                        current_gesture['positions'],
                        current_gesture['velocities'],
                        current_gesture['timestamps']
                    )
                status_label.setText("Gesture completed!")

    # Record gesture data
    if gesture_recording:
        current_gesture['timestamps'].append(t_sec)
        current_gesture['positions'].append(position.copy())
        current_gesture['velocities'].append(velocity.copy())
        current_gesture['orientations'].append([roll, pitch, yaw])
        current_gesture['accelerations'].append([ax, ay, az])
        current_gesture['angular_velocities'].append([gx, gy, gz])
        
        # Limit gesture length
        if len(current_gesture['timestamps']) > MAX_GESTURE_POINTS:
            for key in current_gesture:
                current_gesture[key].pop(0)

    # Update buffers
    t_buf.append(t_sec)
    ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
    gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)
    roll_buf.append(roll); pitch_buf.append(pitch); yaw_buf.append(yaw)
    vx_buf.append(velocity[0]); vy_buf.append(velocity[1]); vz_buf.append(velocity[2])
    pos_x_buf.append(position[0]); pos_y_buf.append(position[1]); pos_z_buf.append(position[2])
    motion_mag_buf.append(motion_mag)
    gesture_state_buf.append(gesture_state)

    # Update UI
    if len(t_buf) > 0:
        # Get recent data for plotting
        recent_window = min(int(3.0 * SAMPLE_HZ), len(t_buf))
        t_recent = list(t_buf)[-recent_window:]
        
        # Update threshold lines
        motion_threshold_line.setPos(MOTION_THRESH)
        stillness_threshold_line.setPos(STILLNESS_THRESH)
        
        # Update plots with recent data
        ax_recent = list(ax_buf)[-recent_window:]
        ay_recent = list(ay_buf)[-recent_window:]
        az_recent = list(az_buf)[-recent_window:]
        gx_recent = list(gx_buf)[-recent_window:]
        gy_recent = list(gy_buf)[-recent_window:]
        gz_recent = list(gz_buf)[-recent_window:]
        roll_recent = list(roll_buf)[-recent_window:]
        pitch_recent = list(pitch_buf)[-recent_window:]
        yaw_recent = list(yaw_buf)[-recent_window:]
        vx_recent = list(vx_buf)[-recent_window:]
        vy_recent = list(vy_buf)[-recent_window:]
        vz_recent = list(vz_buf)[-recent_window:]
        motion_mag_recent = list(motion_mag_buf)[-recent_window:]
        gesture_state_recent = list(gesture_state_buf)[-recent_window:]
        pos_x_recent = list(pos_x_buf)[-recent_window:]
        pos_y_recent = list(pos_y_buf)[-recent_window:]
        pos_z_recent = list(pos_z_buf)[-recent_window:]

        # Update curves
        curve_ax.setData(t_recent, ax_recent); curve_ay.setData(t_recent, ay_recent); curve_az.setData(t_recent, az_recent)
        curve_gx.setData(t_recent, gx_recent); curve_gy.setData(t_recent, gy_recent); curve_gz.setData(t_recent, gz_recent)
        curve_motion_mag.setData(t_recent, motion_mag_recent)
        curve_vx.setData(t_recent, vx_recent); curve_vy.setData(t_recent, vy_recent); curve_vz.setData(t_recent, vz_recent)
        curve_roll.setData(t_recent, roll_recent); curve_pitch.setData(t_recent, pitch_recent); curve_yaw.setData(t_recent, yaw_recent)
        curve_gesture_state.setData(t_recent, gesture_state_recent)
        curve_xy_current.setData(pos_x_recent, pos_y_recent)
        curve_xz_current.setData(pos_x_recent, pos_z_recent)

        # Update gesture visualization
        if current_gesture['positions']:
            gesture_pos = np.array(current_gesture['positions'])
            curve_xy_gesture.setData(gesture_pos[:, 0], gesture_pos[:, 1])
            curve_xz_gesture.setData(gesture_pos[:, 0], gesture_pos[:, 2])
            
            # Mark start and end points
            if len(gesture_pos) > 1:
                scatter_start.setData(pos=[(gesture_pos[0, 0], gesture_pos[0, 1])])
                scatter_end.setData(pos=[(gesture_pos[-1, 0], gesture_pos[-1, 1])])
            
            # Calculate and show speed profile
            if len(current_gesture['velocities']) > 1:
                velocities = np.array(current_gesture['velocities'])
                speeds = np.linalg.norm(velocities, axis=1)
                gesture_times = np.array(current_gesture['timestamps']) - current_gesture['timestamps'][0]
                curve_speed_profile.setData(gesture_times, speeds)

        # Update gesture regions
        if gesture_recording and current_gesture['timestamps']:
            start_time = current_gesture['timestamps'][0]
            end_time = current_gesture['timestamps'][-1]
            gesture_region_acc.setRegion([start_time, end_time])
            gesture_region_gyro.setRegion([start_time, end_time])

        # Set plot ranges
        t_min = t_recent[0] if t_recent else t_sec - 3.0
        for p in [p1, p2, p3, p4, p5, p6]:
            p.setXRange(t_min, t_sec, padding=0)

    # Update info labels
    if current_gesture['timestamps']:
        duration = current_gesture['timestamps'][-1] - current_gesture['timestamps'][0]
        gesture_info_label.setText(f"Gesture: {len(current_gesture['timestamps'])} points, {duration:.2f}s")
    else:
        gesture_info_label.setText("Gesture: None")

    metrics_text = f"Path: {gesture_metrics['path_length']:.3f}m, Max Speed: {gesture_metrics['max_speed']:.2f}m/s, Displacement: {gesture_metrics['displacement']:.3f}m"
    metrics_label.setText(metrics_text)

# ---------------- Timer ----------------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)  # 100Hz update rate

# Show window
win.show()
app.exec()