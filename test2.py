# final_imu_position.py - Enhanced IMU Position Tracking System
import serial, math, numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtGui import QMatrix4x4
from ahrs.filters import Madgwick
from collections import deque
import time

# ---------------- USER CONFIG ----------------
PORT = 'COM7'           # Adjust to your port
BAUD = 115200
GYRO_IN_DEG = True      # Arduino outputs deg/s
BETA = 0.25
INITIAL_SAMPLE_HZ = 100.0
MAX_POINTS = 400

# Enhanced ZUPT parameters
ZUPT_ACCEL_THRESH = 0.15   # m/s^2 tolerance from stationary
ZUPT_GYRO_THRESH = 1.0     # deg/s gyro threshold
ZUPT_MIN_DURATION = 0.1    # minimum stationary duration in seconds
ZUPT_CONFIDENCE_WINDOW = 10 # samples to check for stationary state

# Kalman filter parameters for position tracking
PROCESS_NOISE_ACC = 0.1    # process noise for acceleration
PROCESS_NOISE_VEL = 0.05   # process noise for velocity
MEASUREMENT_NOISE = 0.5    # measurement noise

# Drift correction parameters
DRIFT_CORRECTION_ALPHA = 0.98  # for exponential moving average
BIAS_LEARNING_RATE = 0.001     # how fast to adapt to sensor bias
MAX_VELOCITY = 5.0             # maximum reasonable velocity (m/s)
VELOCITY_DECAY = 0.99          # velocity decay factor when not moving
# ---------------------------------------------

class KalmanFilter:
    """Simple 1D Kalman filter for position/velocity estimation"""
    def __init__(self, process_noise_pos=0.01, process_noise_vel=0.1, measurement_noise=0.5):
        # State: [position, velocity]
        self.x = np.array([0.0, 0.0])  # initial state
        self.P = np.eye(2) * 10        # initial covariance
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1.0, 0.0],
                          [0.0, 1.0]])  # dt will be set dynamically
        
        # Process noise covariance
        self.Q = np.array([[process_noise_pos, 0.0],
                          [0.0, process_noise_vel]])
        
        # Measurement matrix (we observe acceleration)
        self.H = np.array([[0.0, 0.0]])  # will be updated for velocity reset
        
        # Measurement noise
        self.R = np.array([[measurement_noise]])
    
    def predict(self, dt, acceleration):
        """Predict next state using acceleration input"""
        # Update state transition matrix with dt
        self.F[0, 1] = dt
        
        # Control input matrix for acceleration
        B = np.array([[0.5 * dt * dt], [dt]])
        
        # Predict state
        self.x = self.F @ self.x + B.flatten() * acceleration
        
        # Update process noise with dt
        Q_dt = np.array([[0.25 * dt**4, 0.5 * dt**3],
                        [0.5 * dt**3, dt**2]]) * PROCESS_NOISE_ACC
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + Q_dt
    
    def update_velocity_zero(self):
        """Update with zero velocity measurement (ZUPT)"""
        # Measurement matrix for velocity
        H = np.array([[0.0, 1.0]])
        
        # Innovation
        y = 0.0 - H @ self.x
        S = H @ self.P @ H.T + MEASUREMENT_NOISE
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # Update state and covariance
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
    
    def get_position(self):
        return self.x[0]
    
    def get_velocity(self):
        return self.x[1]

class AccelerationBiasEstimator:
    """Estimates and corrects for acceleration bias"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.acc_buffer = deque(maxlen=window_size)
        self.bias = np.zeros(3)
        self.is_stationary_buffer = deque(maxlen=window_size)
    
    def update(self, acceleration, is_stationary):
        self.acc_buffer.append(acceleration.copy())
        self.is_stationary_buffer.append(is_stationary)
        
        if len(self.acc_buffer) == self.window_size:
            # Calculate bias only during stationary periods
            stationary_indices = [i for i, stat in enumerate(self.is_stationary_buffer) if stat]
            if len(stationary_indices) > self.window_size // 4:  # need enough stationary samples
                stationary_acc = np.array([self.acc_buffer[i] for i in stationary_indices])
                measured_bias = np.mean(stationary_acc, axis=0)
                # Smooth bias update
                self.bias = DRIFT_CORRECTION_ALPHA * self.bias + (1 - DRIFT_CORRECTION_ALPHA) * measured_bias
    
    def get_corrected_acceleration(self, acceleration):
        return acceleration - self.bias

class ZUPTDetector:
    """Enhanced Zero Velocity Update detector"""
    def __init__(self):
        self.stationary_count = 0
        self.last_stationary_time = 0
        self.confidence_buffer = deque(maxlen=ZUPT_CONFIDENCE_WINDOW)
    
    def is_stationary(self, acceleration, gyroscope, current_time):
        # Check acceleration magnitude (should be ~9.81 m/s² when stationary)
        acc_magnitude = np.linalg.norm(acceleration)
        acc_stationary = abs(acc_magnitude - 9.81) < ZUPT_ACCEL_THRESH
        
        # Check gyroscope magnitude
        gyro_magnitude = np.linalg.norm(gyroscope)
        gyro_stationary = gyro_magnitude < ZUPT_GYRO_THRESH
        
        # Current sample assessment
        currently_stationary = acc_stationary and gyro_stationary
        self.confidence_buffer.append(currently_stationary)
        
        # Require confidence over multiple samples
        if len(self.confidence_buffer) == ZUPT_CONFIDENCE_WINDOW:
            confidence_ratio = sum(self.confidence_buffer) / len(self.confidence_buffer)
            confident_stationary = confidence_ratio > 0.8
        else:
            confident_stationary = False
        
        # Require minimum duration of stationary state
        if confident_stationary:
            if self.stationary_count == 0:
                self.last_stationary_time = current_time
            self.stationary_count += 1
            duration = current_time - self.last_stationary_time
            return duration >= ZUPT_MIN_DURATION
        else:
            self.stationary_count = 0
            return False

# Initialize serial connection
print("Initializing IMU Position Tracking System...")
print(f"Connecting to {PORT} at {BAUD} baud...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=0)
    time.sleep(2.0)  # Give Arduino time to initialize
    print("✓ Serial connection established")
except Exception as e:
    print(f"✗ Failed to connect to {PORT}: {e}")
    print("Please check your port settings and try again.")
    exit(1)

# GUI setup
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Enhanced IMU Position Tracking System")
win.resize(1400, 900)

# Initialize tracking variables
origin_set = False
origin_pos = np.zeros(3)

# Create layout - 2x3 grid for comprehensive visualization
# Top row: Raw data and corrections
p1 = win.addPlot(title="Acceleration (m/s²) - Raw vs Bias-Corrected"); p1.addLegend()
curve_ax_raw = p1.plot(pen='r', name='ax_raw')
curve_ay_raw = p1.plot(pen='g', name='ay_raw') 
curve_az_raw = p1.plot(pen='b', name='az_raw')
curve_ax_corr = p1.plot(pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2), name='ax_corrected')
curve_ay_corr = p1.plot(pen=pg.mkPen('g', style=QtCore.Qt.DashLine, width=2), name='ay_corrected')
curve_az_corr = p1.plot(pen=pg.mkPen('b', style=QtCore.Qt.DashLine, width=2), name='az_corrected')

p2 = win.addPlot(title="Velocity (m/s) - Kalman Filtered"); p2.addLegend()
curve_vx = p2.plot(pen='r', name='vx')
curve_vy = p2.plot(pen='g', name='vy')
curve_vz = p2.plot(pen='b', name='vz')

p3 = win.addPlot(title="Data Quality & ZUPT Events"); p3.addLegend()
curve_quality = p3.plot(pen='orange', name='Quality Score')

# Second row: Position tracking results
win.nextRow()
p4 = win.addPlot(title="Enhanced XY Trajectory (Kalman + ZUPT + Bias Correction)")
p4.setAspectLocked(True)
p4.setLabel('left', 'Y Position', 'm')
p4.setLabel('bottom', 'X Position', 'm')
curve_xy_enhanced = p4.plot(pen=pg.mkPen('cyan', width=3), name='Enhanced Path')
zupt_scatter = p4.plot(pen=None, symbol='o', symbolBrush='red', symbolSize=6, name='ZUPT Points')
start_marker = p4.plot(pen=None, symbol='s', symbolBrush='green', symbolSize=10, name='Start')

p5 = win.addPlot(title="Method Comparison - XY Trajectory")
p5.setAspectLocked(True)  
p5.setLabel('left', 'Y Position', 'm')
p5.setLabel('bottom', 'X Position', 'm')
p5.addLegend()
curve_xy_raw_comp = p5.plot(pen=pg.mkPen('yellow', width=2), name='Raw Integration')
curve_xy_enhanced_comp = p5.plot(pen=pg.mkPen('cyan', width=2), name='Enhanced Method')

p6 = win.addPlot(title="Position Error Metrics")
p6.addLegend()
curve_pos_drift = p6.plot(pen='red', name='Position Drift (m)')
curve_vel_magnitude = p6.plot(pen='blue', name='Velocity Magnitude (m/s)')

# 3D Orientation visualization
w3d = gl.GLViewWidget()
w3d.setWindowTitle('IMU Orientation - Real-time 3D View')
w3d.setCameraPosition(distance=10, azimuth=30, elevation=20)

# Add coordinate axes
axis = gl.GLAxisItem()
axis.setSize(3,3,3)
w3d.addItem(axis)

# Add grid
grid = gl.GLGridItem()
grid.scale(2,2,1)
w3d.addItem(grid)

# Create orientation cube
verts = np.array([[1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],
                  [-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1]])
faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [1,2,6],[1,6,5],[0,3,7],[0,7,4]])
colors = np.array([[1,0,0,0.6]]*2 + [[0,1,0,0.6]]*2 +
                 [[0,0,1,0.6]]*2 + [[1,1,0,0.6]]*2 +
                 [[0,1,1,0.6]]*2 + [[1,0,1,0.6]]*2)
meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
cube = gl.GLMeshItem(meshdata=meshdata, smooth=False,
                     drawEdges=True, edgeColor=(1,1,1,1))
cube.scale(2,2,2)
w3d.addItem(cube)
w3d.show()
win.show()

# Initialize processing components
madgwick = Madgwick(beta=BETA, sampleperiod=1.0/INITIAL_SAMPLE_HZ)
kalman_x = KalmanFilter()
kalman_y = KalmanFilter()  
kalman_z = KalmanFilter()
bias_estimator = AccelerationBiasEstimator()
zupt_detector = ZUPTDetector()

# State variables
q = np.array([1.0, 0.0, 0.0, 0.0])
last_time = None
raw_velocity = np.zeros(3)
raw_position = np.zeros(3)

# Data buffers for visualization
t_buf = []
ax_raw_buf, ay_raw_buf, az_raw_buf = [], [], []
ax_corr_buf, ay_corr_buf, az_corr_buf = [], [], []
vx_buf, vy_buf, vz_buf = [], [], []
px_enhanced_buf, py_enhanced_buf = [], []
px_raw_buf, py_raw_buf = [], []
zupt_x_buf, zupt_y_buf = [], []
quality_buf = []
pos_drift_buf = []
vel_mag_buf = []
start_marked = False

# Statistics tracking
total_samples = 0
zupt_count = 0
max_velocity = 0

# Helper functions
def quat_to_euler_deg(q):
    """Convert quaternion to Euler angles in degrees"""
    w, x, y, z = q
    roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    s = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.degrees(math.asin(s))
    yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw

def cube_set_quat(q):
    """Update 3D cube orientation from quaternion"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x), 0],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y), 0],
        [0,0,0,1]
    ], dtype=float)
    cube.resetTransform()
    cube.setTransform(QMatrix4x4(*R.flatten()))

def rotate_sensor_to_world(q, ax, ay, az):
    """Transform acceleration from sensor frame to world frame"""
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ])
    return R.dot(np.array([ax, ay, az]))

def calculate_quality_score(status):
    """Convert status string to numeric quality score for plotting"""
    if "EXCELLENT" in status:
        return 4.0
    elif "GOOD" in status:
        return 3.0  
    elif "FAIR" in status:
        return 2.0
    elif "POOR" in status:
        return 1.0
    else:
        return 0.0

def update():
    """Main update loop - processes new IMU data"""
    global last_time, q, raw_velocity, raw_position, origin_set, origin_pos
    global total_samples, zupt_count, max_velocity, start_marked

    # Read latest data from serial
    line = None
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            line = None
            break
    if not line: 
        return

    # Parse CSV data
    parts = line.split(',')
    if len(parts) < 7: 
        return
    try:
        t_ms, ax, ay, az, gx, gy, gz = map(float, parts[:7])
        status = parts[7].strip() if len(parts) > 7 else "UNKNOWN"
        
        # Skip invalid readings flagged by Arduino
        if "INVALID" in status:
            return
            
    except:
        return

    total_samples += 1

    # Time management
    if last_time is None:
        last_time = t_ms
        return
    dt = (t_ms - last_time) / 1000.0
    if dt <= 0 or dt > 1.0:  # Skip unrealistic time deltas
        last_time = t_ms
        return
    last_time = t_ms
    current_time = t_ms / 1000.0

    # Update orientation using Madgwick filter
    madgwick.sampleperiod = dt
    g = np.radians([gx, gy, gz]) if GYRO_IN_DEG else np.array([gx, gy, gz])
    a_sensor = np.array([ax, ay, az])
    q = madgwick.updateIMU(q, gyr=g, acc=a_sensor)

    # Transform acceleration to world frame and remove gravity
    a_world = rotate_sensor_to_world(q, ax, ay, az)
    a_motion = a_world - np.array([0, 0, 9.81])

    # ZUPT detection
    is_stationary = zupt_detector.is_stationary(a_world, [gx, gy, gz], current_time)
    if is_stationary:
        zupt_count += 1

    # Bias estimation and correction
    bias_estimator.update(a_motion, is_stationary)
    a_corrected = bias_estimator.get_corrected_acceleration(a_motion)

    # Apply reasonable limits to prevent runaway values
    a_corrected = np.clip(a_corrected, -50, 50)

    # Kalman filter prediction step
    kalman_x.predict(dt, a_corrected[0])
    kalman_y.predict(dt, a_corrected[1])
    kalman_z.predict(dt, a_corrected[2])

    # Apply ZUPT correction when stationary
    if is_stationary:
        kalman_x.update_velocity_zero()
        kalman_y.update_velocity_zero()
        kalman_z.update_velocity_zero()
        
        # Record ZUPT event for visualization
        zupt_x_buf.append(kalman_x.get_position())
        zupt_y_buf.append(kalman_y.get_position())

    # Get enhanced position and velocity estimates
    enhanced_pos = np.array([kalman_x.get_position(), 
                           kalman_y.get_position(), 
                           kalman_z.get_position()])
    enhanced_vel = np.array([kalman_x.get_velocity(), 
                           kalman_y.get_velocity(), 
                           kalman_z.get_velocity()])

    # Raw integration for comparison (with some basic drift reduction)
    if not is_stationary:
        raw_velocity += a_motion * dt
        raw_velocity *= VELOCITY_DECAY
    else:
        raw_velocity *= 0.8  # More aggressive decay when stationary
    
    raw_velocity = np.clip(raw_velocity, -MAX_VELOCITY, MAX_VELOCITY)
    raw_position += raw_velocity * dt

    # Set origin on first stable reading
    # Set origin on first stable reading
    if not origin_set:
        origin_pos = np.array(enhanced_pos, dtype=float).flatten()
        origin_set = True
        print("✓ Origin set. Starting position tracking...")

# Calculate relative positions (always 1D vectors)
    enhanced_pos_rel = np.array(enhanced_pos, dtype=float).flatten() - origin_pos
    raw_pos_rel = np.array(raw_position, dtype=float).flatten() - origin_pos

    # Update statistics
    vel_magnitude = np.linalg.norm(enhanced_vel)
    max_velocity = max(max_velocity, vel_magnitude)
    position_drift = np.linalg.norm(enhanced_pos_rel)

    # Store data for visualization
    t_buf.append(current_time)
    ax_raw_buf.append(a_motion[0])
    ay_raw_buf.append(a_motion[1])
    az_raw_buf.append(a_motion[2])
    ax_corr_buf.append(a_corrected[0])
    ay_corr_buf.append(a_corrected[1])
    az_corr_buf.append(a_corrected[2])
    vx_buf.append(enhanced_vel[0])
    vy_buf.append(enhanced_vel[1])
    vz_buf.append(enhanced_vel[2])
    px_enhanced_buf.append(enhanced_pos_rel[0])
    py_enhanced_buf.append(enhanced_pos_rel[1])
    px_raw_buf.append(raw_pos_rel[0])
    py_raw_buf.append(raw_pos_rel[1])
    quality_buf.append(calculate_quality_score(status))
    pos_drift_buf.append(position_drift)
    vel_mag_buf.append(vel_magnitude)

    # Mark start position
    if not start_marked and len(px_enhanced_buf) > 1:
        start_marker.setData([px_enhanced_buf[0]], [py_enhanced_buf[0]])
        start_marked = True

    # Limit buffer sizes
    if len(t_buf) > MAX_POINTS:
        for buf in [t_buf, ax_raw_buf, ay_raw_buf, az_raw_buf, ax_corr_buf, ay_corr_buf, az_corr_buf,
                   vx_buf, vy_buf, vz_buf, px_enhanced_buf, py_enhanced_buf, px_raw_buf, py_raw_buf,
                   quality_buf, pos_drift_buf, vel_mag_buf]:
            buf.pop(0)

    # Limit ZUPT visualization points
    if len(zupt_x_buf) > 50:
        zupt_x_buf.pop(0)
        zupt_y_buf.pop(0)

    # Update all plots
    curve_ax_raw.setData(t_buf, ax_raw_buf)
    curve_ay_raw.setData(t_buf, ay_raw_buf)
    curve_az_raw.setData(t_buf, az_raw_buf)
    curve_ax_corr.setData(t_buf, ax_corr_buf)
    curve_ay_corr.setData(t_buf, ay_corr_buf)
    curve_az_corr.setData(t_buf, az_corr_buf)
    curve_vx.setData(t_buf, vx_buf)
    curve_vy.setData(t_buf, vy_buf)
    curve_vz.setData(t_buf, vz_buf)
    curve_quality.setData(t_buf, quality_buf)
    curve_xy_enhanced.setData(px_enhanced_buf, py_enhanced_buf)
    curve_xy_raw_comp.setData(px_raw_buf, py_raw_buf)
    curve_xy_enhanced_comp.setData(px_enhanced_buf, py_enhanced_buf)
    curve_pos_drift.setData(t_buf, pos_drift_buf)
    curve_vel_magnitude.setData(t_buf, vel_mag_buf)
    zupt_scatter.setData(zupt_x_buf, zupt_y_buf)

    # Update 3D orientation
    cube_set_quat(q)
    
    # Dynamic trajectory coloring based on quality
    if "EXCELLENT" in status:
        curve_xy_enhanced.setPen(pg.mkPen('cyan', width=3))
    elif "GOOD" in status:
        curve_xy_enhanced.setPen(pg.mkPen('green', width=3))
    elif "FAIR" in status:
        curve_xy_enhanced.setPen(pg.mkPen('yellow', width=3))
    else:
        curve_xy_enhanced.setPen(pg.mkPen('red', width=3))
    
    # Update window title with comprehensive stats
    zupt_rate = (zupt_count / total_samples * 100) if total_samples > 0 else 0
    win.setWindowTitle(f"IMU Position Tracking | Status: {status} | "
                      f"ZUPT: {zupt_rate:.1f}% | Max Vel: {max_velocity:.2f} m/s | "
                      f"Samples: {total_samples}")

# Print startup information
print("\n" + "="*60)
print("🚀 ENHANCED IMU POSITION TRACKING SYSTEM")
print("="*60)
print("Features:")
print("  • Kalman filtering for optimal state estimation")
print("  • Enhanced ZUPT with confidence-based detection")
print("  • Dynamic bias estimation and correction")
print("  • Real-time data quality monitoring")
print("  • Comprehensive visualization suite")
print("  • Arduino calibration integration")
print("\nControls:")
print("  • Wait for Arduino calibration to complete")
print("  • Green square shows trajectory start point")
print("  • Red dots indicate ZUPT (stationary) events") 
print("  • Trajectory color indicates data quality")
print("  • Monitor velocity and drift in real-time")
print("="*60)

# Start the update timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(12)  # ~80 Hz update rate

if __name__ == '__main__':
    try:
        print("✓ Starting GUI... Press Ctrl+C to exit")
        app.exec_()
    except KeyboardInterrupt:
        print("\n⏹ Shutting down...")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("✓ Serial connection closed")
        print("✓ System shutdown complete")