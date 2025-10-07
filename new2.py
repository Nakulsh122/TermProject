# enhanced_gesture_tracking_imu.py
import serial
import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from ahrs.filters import Madgwick
import time
import json
import os
from collections import deque
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------- CONFIG ----------------
PORT = 'COM7'
BAUD = 115200
GYRO_IN_DEG = True
SAMPLE_HZ = 100.0
GESTURE_WINDOW = 1.25  # seconds
MAX_GESTURE_POINTS = int(GESTURE_WINDOW * SAMPLE_HZ)

# Gesture-specific parameters (from better previous version)
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

# Data storage settings
GESTURES_FOLDER = "gesture_data"
METADATA_FILE = "gesture_metadata.json"

# Create folders if they don't exist
if not os.path.exists(GESTURES_FOLDER):
    os.makedirs(GESTURES_FOLDER)

# ---------------- SERIAL ----------------
try:
    ser = serial.Serial(PORT, BAUD, timeout=0)
    time.sleep(1.0)
    serial_connected = True
except:
    print("Serial connection failed - running in demo mode")
    serial_connected = False
    ser = None

# ---------------- GUI ----------------
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("Enhanced Gesture Tracking with Labeling")
win.resize(1800, 1100)
main_layout = QtWidgets.QVBoxLayout(win)

# ---------------- Enhanced Controls ----------------
control_layout = QtWidgets.QHBoxLayout()
main_layout.addLayout(control_layout)

# Gesture detection and labeling controls
gesture_group = QtWidgets.QGroupBox("Gesture Recording & Labeling")
gesture_layout = QtWidgets.QVBoxLayout(gesture_group)

# Label input section
label_layout = QtWidgets.QHBoxLayout()
label_input = QtWidgets.QLineEdit()
label_input.setPlaceholderText("Enter gesture label (e.g., 'wave', 'circle', 'swipe')")
label_input.setFixedWidth(250)
label_layout.addWidget(QtWidgets.QLabel("Gesture Label:"))
label_layout.addWidget(label_input)
gesture_layout.addLayout(label_layout)

# Button layout
button_layout = QtWidgets.QHBoxLayout()
start_gesture_btn = QtWidgets.QPushButton("Start Recording")
stop_gesture_btn = QtWidgets.QPushButton("Stop Recording")
clear_gesture_btn = QtWidgets.QPushButton("Clear Gesture")
save_gesture_btn = QtWidgets.QPushButton("Save Gesture")
auto_detect_cb = QtWidgets.QCheckBox("Auto Detect")
auto_detect_cb.setChecked(True)

# Enhanced button styling
start_gesture_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
stop_gesture_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
save_gesture_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

button_layout.addWidget(start_gesture_btn)
button_layout.addWidget(stop_gesture_btn)
button_layout.addWidget(clear_gesture_btn)
button_layout.addWidget(save_gesture_btn)
button_layout.addWidget(auto_detect_cb)
gesture_layout.addLayout(button_layout)

control_layout.addWidget(gesture_group)

# Parameter controls (from better version)
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

# Dataset management section
dataset_group = QtWidgets.QGroupBox("Dataset Management")
dataset_layout = QtWidgets.QVBoxLayout(dataset_group)

dataset_info_label = QtWidgets.QLabel("Dataset: 0 gestures")
dataset_buttons_layout = QtWidgets.QHBoxLayout()
load_dataset_btn = QtWidgets.QPushButton("Load Dataset")
export_dataset_btn = QtWidgets.QPushButton("Export Dataset")
clear_dataset_btn = QtWidgets.QPushButton("Clear Dataset")

dataset_buttons_layout.addWidget(load_dataset_btn)
dataset_buttons_layout.addWidget(export_dataset_btn) 
dataset_buttons_layout.addWidget(clear_dataset_btn)

dataset_layout.addWidget(dataset_info_label)
dataset_layout.addLayout(dataset_buttons_layout)
control_layout.addWidget(dataset_group)

def get_slider_values():
    return [scale_slider_value(s.value(), min_val, max_val) 
            for s, (min_val, max_val) in zip(sliders, slider_ranges)]

def update_labels():
    for name, s, lbl, (min_val, max_val) in zip(slider_names, sliders, labels, slider_ranges):
        val = scale_slider_value(s.value(), min_val, max_val)
        lbl.setText(f"{name}: {val:.3f}")

for s in sliders:
    s.valueChanged.connect(update_labels)

# Status and metrics (enhanced)
status_layout = QtWidgets.QHBoxLayout()
status_label = QtWidgets.QLabel("Status: Ready for gesture")
status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
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

# ---------------- Data Structures (from better version) ----------------
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

# Enhanced gesture recording with labeling
gesture_recording = False
gesture_data = []
gesture_start_time = None
gesture_end_time = None
current_gesture = {
    'label': '',
    'timestamps': [],
    'positions': [],
    'velocities': [],
    'orientations': [],
    'accelerations': [],
    'angular_velocities': []
}

# Filter states (from better version)
madgwick = Madgwick(beta=BETA_DEFAULT, sampleperiod=1.0/SAMPLE_HZ)
q = np.array([1.0, 0.0, 0.0, 0.0])
last_time = None

# Motion tracking (better implementation)
velocity = np.zeros(3)
position = np.zeros(3)
prev_accel_world = np.zeros(3)

# Gesture detection state (from better version)
gesture_state = 0  # 0: idle, 1: detecting, 2: recording
motion_start_time = None
last_significant_motion = None

# Calibration
accel_bias = np.zeros(3)
gyro_bias = np.zeros(3)

# Enhanced dataset management
gesture_counter = {}  # Track count per label
total_gestures = 0

# Gesture metrics
gesture_metrics = {
    'duration': 0,
    'path_length': 0,
    'max_speed': 0,
    'avg_speed': 0,
    'displacement': 0
}

# Load existing dataset metadata
def load_gesture_metadata():
    global gesture_counter, total_gestures
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                gesture_counter = data.get('counter', {})
                total_gestures = data.get('total', 0)
    except:
        gesture_counter = {}
        total_gestures = 0

def save_gesture_metadata():
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump({
                'counter': gesture_counter,
                'total': total_gestures,
                'last_updated': time.time()
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

load_gesture_metadata()

# ---------------- Gesture Functions (from better version) ----------------
def calculate_enhanced_motion_magnitude(ax, ay, az, gx, gy, gz, q):
    """Enhanced motion magnitude that's orientation-independent"""
    # Calculate acceleration in world frame first
    def rotate_vector_by_quaternion(v, q):
        w, x, y, z = q
        qv = np.array([0, v[0], v[1], v[2]])
        temp = np.array([
            -x*qv[1] - y*qv[2] - z*qv[3],
            w*qv[1] + y*qv[3] - z*qv[2],
            w*qv[2] + z*qv[1] - x*qv[3],
            w*qv[3] + x*qv[2] - y*qv[1]
        ])
        result = np.array([
            temp[0]*w + temp[1]*x + temp[2]*y + temp[3]*z,
            -temp[0]*x + temp[1]*w - temp[2]*z + temp[3]*y,
            -temp[0]*y + temp[1]*z + temp[2]*w - temp[3]*x,
            -temp[0]*z - temp[1]*y + temp[2]*x + temp[3]*w
        ])
        return result[1:4]
    
    # Transform acceleration to world frame
    accel_world = rotate_vector_by_quaternion(np.array([ax, ay, az]), q)
    
    # Remove gravity component
    gravity_sensor = np.array([0, 0, 1])
    gravity_world = rotate_vector_by_quaternion(gravity_sensor, q) * 9.81
    accel_motion = accel_world - gravity_world
    
    # Calculate motion magnitude using world-frame motion
    accel_mag = np.linalg.norm(accel_motion)
    gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)
    
    # Weight both components
    return accel_mag * 0.7 + gyro_mag * 0.3

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
        'label': '',
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
    """Enhanced save with labeling system"""
    global total_gestures, gesture_counter
    
    if not current_gesture['timestamps']:
        status_label.setText("No gesture data to save")
        status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        return
    
    # Get label from input field
    label = label_input.text().strip()
    if not label:
        status_label.setText("Error: Please enter a gesture label!")
        status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        return
    
    # Update counter
    if label not in gesture_counter:
        gesture_counter[label] = 0
    gesture_counter[label] += 1
    total_gestures += 1
    
    # Create filename with incremental counter
    filename = f"{label}_{gesture_counter[label]:03d}"
    
    try:
        # Save as CSV
        csv_path = os.path.join(GESTURES_FOLDER, f"{filename}.csv")
        with open(csv_path, 'w') as f:
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
        
        # Save as pickle for ML
        pkl_path = os.path.join(GESTURES_FOLDER, f"{filename}.pkl")
        gesture_data_full = {
            'label': label,
            'filename': filename,
            'timestamp': time.time(),
            'data': current_gesture,
            'metrics': gesture_metrics,
            'sample_rate': SAMPLE_HZ
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(gesture_data_full, f)
        
        # Save metadata
        save_gesture_metadata()
        
        status_label.setText(f"Saved: {filename}")
        status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        
        # Update dataset display
        update_dataset_info()
        
        # Clear input for next gesture
        label_input.clear()
        
    except Exception as e:
        status_label.setText(f"Error saving: {str(e)}")
        status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

def update_dataset_info():
    unique_labels = len(gesture_counter)
    dataset_info_label.setText(f"Dataset: {total_gestures} gestures, {unique_labels} unique labels")

def export_dataset():
    """Export complete dataset"""
    try:
        export_data = {
            'metadata': {
                'total_gestures': total_gestures,
                'gesture_counter': gesture_counter,
                'export_time': time.time(),
                'sample_rate': SAMPLE_HZ
            },
            'gestures': []
        }
        
        # Load all gesture files
        for filename in os.listdir(GESTURES_FOLDER):
            if filename.endswith('.pkl'):
                pkl_path = os.path.join(GESTURES_FOLDER, filename)
                with open(pkl_path, 'rb') as f:
                    gesture_data = pickle.load(f)
                    export_data['gestures'].append(gesture_data)
        
        # Save complete dataset
        export_path = f"complete_dataset_{int(time.time())}.pkl"
        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        status_label.setText(f"Dataset exported: {export_path}")
        status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        
    except Exception as e:
        status_label.setText(f"Export error: {str(e)}")
        status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

def start_manual_recording():
    """Start manual gesture recording"""
    global gesture_recording, gesture_start_time
    if not label_input.text().strip():
        status_label.setText("Error: Please enter a gesture label first!")
        status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        return
        
    gesture_recording = True
    gesture_start_time = time.time()
    reset_gesture()
    current_gesture['label'] = label_input.text().strip()
    status_label.setText("Recording gesture manually...")
    status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

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
    status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")

# Connect buttons
start_gesture_btn.clicked.connect(start_manual_recording)
stop_gesture_btn.clicked.connect(stop_manual_recording)
clear_gesture_btn.clicked.connect(reset_gesture)
save_gesture_btn.clicked.connect(save_gesture_data)
export_dataset_btn.clicked.connect(export_dataset)

# Initialize dataset display
update_dataset_info()

# ---------------- Enhanced Update Loop (from better version) ----------------
def update():
    global last_time, q, velocity, position, prev_accel_world
    global gesture_recording, gesture_state, motion_start_time, last_significant_motion
    global accel_bias, gyro_bias, gesture_metrics

    # Handle demo mode
    if not serial_connected:
        t_ms = time.time() * 1000
        ax_raw = np.random.normal(0, 0.1) + 0.5 * np.sin(time.time())
        ay_raw = np.random.normal(0, 0.1) + 0.3 * np.cos(time.time() * 1.2)
        az_raw = 9.81 + np.random.normal(0, 0.1)
        gx_raw = np.random.normal(0, 0.5)
        gy_raw = np.random.normal(0, 0.5) 
        gz_raw = np.random.normal(0, 0.5)
        line = f"{t_ms},{ax_raw},{ay_raw},{az_raw},{gx_raw},{gy_raw},{gz_raw}"
    else:
        # Read real serial data
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

    # Enhanced Euler angle calculation with gimbal lock protection
    def quaternion_to_euler_robust(q):
        """Convert quaternion to Euler angles with gimbal lock handling"""
        w, x, y, z = q
        
        # Normalize quaternion to prevent numerical errors
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation) - handle gimbal lock
        sinp = 2 * (w * y - z * x)
        # Clamp to prevent numerical issues and handle gimbal lock
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    # Use robust Euler calculation
    roll, pitch, yaw = quaternion_to_euler_robust(q)

    # Enhanced robust transformation to handle gimbal lock
    # Use quaternion directly for rotation to avoid gimbal lock issues
    def rotate_vector_by_quaternion(v, q):
        """Rotate vector v by quaternion q - more stable than rotation matrix"""
        w, x, y, z = q
        # Quaternion rotation: q * v * q_conjugate
        qv = np.array([0, v[0], v[1], v[2]])  # Pure quaternion from vector
        
        # q * qv
        temp = np.array([
            -x*qv[1] - y*qv[2] - z*qv[3],
            w*qv[1] + y*qv[3] - z*qv[2],
            w*qv[2] + z*qv[1] - x*qv[3],
            w*qv[3] + x*qv[2] - y*qv[1]
        ])
        
        # temp * q_conjugate
        result = np.array([
            temp[0]*w + temp[1]*x + temp[2]*y + temp[3]*z,
            -temp[0]*x + temp[1]*w - temp[2]*z + temp[3]*y,
            -temp[0]*y + temp[1]*z + temp[2]*w - temp[3]*x,
            -temp[0]*z - temp[1]*y + temp[2]*x + temp[3]*w
        ])
        
        return result[1:4]  # Return only vector part
    
    # Apply quaternion rotation - more stable than matrix
    accel_world = rotate_vector_by_quaternion(np.array([ax, ay, az]), q)
    
    # Robust gravity removal using quaternion-based gravity vector
    gravity_sensor = np.array([0, 0, 1])  # Gravity in sensor frame (down)
    gravity_world = rotate_vector_by_quaternion(gravity_sensor, q) * 9.81
    accel_world = accel_world - gravity_world

    # Enhanced velocity integration with adaptive filtering based on orientation stability
    # Check quaternion stability to detect potential gimbal lock situations
    q_magnitude = np.linalg.norm(q)
    orientation_stable = abs(q_magnitude - 1.0) < 0.1  # Quaternion should have unit magnitude
    
    # Adaptive velocity decay based on orientation stability
    if orientation_stable:
        effective_decay = VEL_DECAY
    else:
        # More aggressive decay when orientation is unstable
        effective_decay = VEL_DECAY * 0.95
        
    # Enhanced integration with outlier rejection
    accel_magnitude = np.linalg.norm(accel_world)
    
    # Reject unrealistic acceleration values that might come from orientation errors
    if accel_magnitude < 50.0:  # Reasonable threshold for human gestures
        velocity = velocity * effective_decay + accel_world * dt
        position = position + velocity * dt
    else:
        # If acceleration seems unrealistic, just apply decay
        velocity = velocity * effective_decay
        position = position + velocity * dt
        
    # Additional smoothing for position when orientation is unstable
    if not orientation_stable:
        # Apply extra smoothing to position
        if hasattr(update, 'prev_position'):
            position = 0.7 * position + 0.3 * update.prev_position
        update.prev_position = position.copy()
    else:
        if not hasattr(update, 'prev_position'):
            update.prev_position = position.copy()

    # Calculate motion magnitude for gesture detection (orientation-independent)
    motion_mag = calculate_enhanced_motion_magnitude(ax, ay, az, gx, gy, gz, q)

    # Auto gesture detection (better implementation)
    if auto_detect_cb.isChecked() and not gesture_recording:
        if gesture_state == 0:  # Idle
            if motion_mag > MOTION_THRESH:
                gesture_state = 1
                motion_start_time = t_sec
                reset_gesture()
                status_label.setText("Motion detected - starting gesture capture...")
                status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        
        elif gesture_state == 1:  # Detecting
            if motion_mag > MOTION_THRESH:
                last_significant_motion = t_sec
            
            # Check if we should start recording
            if last_significant_motion and (t_sec - motion_start_time) > 0.1:
                gesture_state = 2
                gesture_recording = True
                status_label.setText("Recording gesture automatically...")
                status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            
            # Timeout if no significant motion
            if (t_sec - motion_start_time) > 1.0:
                gesture_state = 0
                status_label.setText("No gesture detected - returning to idle")
                status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        
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
                status_label.setText("Gesture completed! Enter label and save.")
                status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")

    # Record gesture data when recording (manual or auto)
    if gesture_recording:
        current_gesture['timestamps'].append(t_sec)
        current_gesture['positions'].append(position.copy())
        current_gesture['velocities'].append(velocity.copy())
        current_gesture['orientations'].append([roll, pitch, yaw])
        current_gesture['accelerations'].append([ax, ay, az])
        current_gesture['angular_velocities'].append([gx, gy, gz])
        
        # Limit gesture length to target duration
        if len(current_gesture['timestamps']) > MAX_GESTURE_POINTS:
            for key in current_gesture:
                if key != 'label' and isinstance(current_gesture[key], list):
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

# Initialize
if serial_connected:
    status_label.setText("Connected - Ready for gesture recording")
else:
    status_label.setText("Demo mode - Ready for gesture recording")

# Show window
win.show()
app.exec()