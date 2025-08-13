"""
imu_final_playback.py

Usage:
    python imu_final_playback.py

Requirements:
    pip install pandas numpy plotly
    # ahrs is optional but recommended for better orientation:
    pip install ahrs

Place your CSV named 'test2.csv' (or change CSV_FILE) next to this script.
CSV must have columns: time, ax, ay, az, wx, wy, wz
 - time: seconds (monotonic)
 - ax,ay,az: accelerometer in m/s^2 (device frame)
 - wx,wy,wz: gyroscope in rad/s (device frame)

Output:
    imu_animation_final.html  (opens in your default browser)
"""

import os
import webbrowser
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from time import perf_counter

pio.renderers.default = "browser"

# ----------------- USER TUNABLE PARAMETERS -----------------
CSV_FILE = r"d:\SEM7\Ubiqutous\TermProject\Testing\test3.csv"
OUT_HTML = "imu_animation_final.html"

# Calibration and filtering / ZUPT settings
CALIB_TIME = 12.0            # seconds of initial stationary data to estimate biases
USE_MADGWICK = True          # set False to skip Madgwick (fallback quaternion integration used)
MADGWICK_BETA = 0.12         # Madgwick beta parameter (tune 0.03 - 0.12)
ZUPT_WINDOW = 0.6            # seconds sliding window for stationary detection
ZUPT_STD_THRESHOLD = 0.10    # accel magnitude std threshold (m/s^2)
GYRO_BIAS_ENABLE = True

# KF/Correction tuning (simple velocity+bias correction)
BIAS_PROCESS_VAR = 1e-7
VEL_PROCESS_VAR = 1e-4
MEAS_VAR_VEL = 1e-3

# Plotting / performance
MAX_FRAMES = 1200            # downsample to at most this many frames for the animation (set None to keep all)
AUTO_ROTATE = True
ROTATE_DEG_PER_FRAME = 0.8
SPEED_SCALE = 1.0            # 1.0 = roughly real-time playback per data timestamps
# -----------------------------------------------------------

# Utility: timed print
def tprint(msg):
    print(f"[{perf_counter():.2f}] {msg}")

# Quaternion helpers (w,x,y,z)
def quat_mul(q, r):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ], dtype=float)

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def quat_from_omega(omega, dt):
    # omega: [wx,wy,wz] rad/s
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = omega / np.linalg.norm(omega)
    w = np.cos(theta/2.0)
    xyz = axis * np.sin(theta/2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)

def quat_to_R(q):
    w,x,y,z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

# Try to import Madgwick (optional)
try:
    if USE_MADGWICK:
        from ahrs.filters import Madgwick
        madgwick = Madgwick(beta=MADGWICK_BETA)
        USE_MADGWICK = True
        tprint("Using ahrs.Madgwick for orientation.")
    else:
        USE_MADGWICK = False
        tprint("Madgwick disabled by configuration.")
except Exception:
    USE_MADGWICK = False
    tprint("ahrs.Madgwick not available — falling back to quaternion integration.")

# Main processing
def main():
    try:
        tprint("Starting IMU final playback script...")

        if not os.path.exists(CSV_FILE):
            raise FileNotFoundError(f"CSV not found: {os.path.abspath(CSV_FILE)}")
        tprint(f"Loading CSV: {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        tprint(f"CSV loaded: {len(df)} rows; columns: {list(df.columns)}")

        # Validate columns
        for c in ("time","ax","ay","az","wx","wy","wz"):
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        # Optionally downsample uniformly to limit frames
        N_total = len(df)
        step = 1
        if MAX_FRAMES and N_total > MAX_FRAMES:
            step = int(np.ceil(N_total / MAX_FRAMES))
            df = df.iloc[::step].reset_index(drop=True)
            tprint(f"Downsampled data by step={step}. New length={len(df)}")

        times = df["time"].to_numpy()
        axs = df["ax"].to_numpy()
        ays = df["ay"].to_numpy()
        azs = df["az"].to_numpy()
        wxs = df["wx"].to_numpy()
        wys = df["wy"].to_numpy()
        wzs = df["wz"].to_numpy()

        # compute dt array
        dts = np.diff(times)
        med_dt = np.median(dts) if len(dts)>0 else 0.02
        dt_array = np.concatenate(([med_dt], dts))
        N = len(times)
        tprint(f"Using {N} samples; median dt = {med_dt:.4f} s")

        # Calibration: use initial CALIB_TIME seconds
        calib_end_idx = np.searchsorted(times, times[0] + CALIB_TIME)
        calib_end_idx = max(3, min(calib_end_idx, N-1))
        tprint(f"Calibration interval: samples [0 .. {calib_end_idx-1}] ({calib_end_idx} samples)")

        gyro_bias = np.array([wxs[:calib_end_idx].mean(), wys[:calib_end_idx].mean(), wzs[:calib_end_idx].mean()])
        tprint(f"Estimated gyro_bias (rad/s): {gyro_bias}")

        acc_calib_mean = np.array([axs[:calib_end_idx].mean(), ays[:calib_end_idx].mean(), azs[:calib_end_idx].mean()])
        tprint(f"Accelerometer mean during calibration (m/s^2): {acc_calib_mean}")

        # Estimate accel bias to align gravity; optional
        accel_bias = acc_calib_mean - np.array([0.0, 0.0, 9.81])
        tprint(f"Estimated accel bias (device frame) to subtract (m/s^2): {accel_bias}")

        # Initial quaternion from accel mean (roll, pitch) with yaw=0
        axm, aym, azm = acc_calib_mean
        init_roll = np.arctan2(-axm, np.sqrt(aym*aym + azm*azm))
        init_pitch = np.arctan2(aym, azm)
        init_yaw = 0.0
        def euler_to_quat(roll, pitch, yaw):
            cr = np.cos(roll/2); sr = np.sin(roll/2)
            cp = np.cos(pitch/2); sp = np.sin(pitch/2)
            cy = np.cos(yaw/2); sy = np.sin(yaw/2)
            w = cr*cp*cy + sr*sp*sy
            x = sr*cp*cy - cr*sp*sy
            y = cr*sp*cy + sr*cp*sy
            z = cr*cp*sy - sr*sp*cy
            return np.array([w, x, y, z], dtype=float)
        q = euler_to_quat(init_roll, init_pitch, init_yaw)
        q = quat_normalize(q)
        tprint(f"Initial quaternion from accel mean (w,x,y,z): {q}")

        # Subtract gyro bias and accel bias from data for processing
        if GYRO_BIAS_ENABLE:
            wxs = wxs - gyro_bias[0]
            wys = wys - gyro_bias[1]
            wzs = wzs - gyro_bias[2]
        axs = axs - accel_bias[0]
        ays = ays - accel_bias[1]
        azs = azs - accel_bias[2]

        # Prepare arrays
        qs = np.zeros((N,4), dtype=float)
        Rs = np.zeros((N,3,3), dtype=float)
        pos = np.zeros((N,3), dtype=float)
        vel = np.zeros((N,3), dtype=float)

        qs[0] = q
        Rs[0] = quat_to_R(q)
        pos[0] = np.zeros(3)
        vel[0] = np.zeros(3)

        # ZUPT detection sliding window
        window_len = max(1, int(round(ZUPT_WINDOW / med_dt)))
        stationary = np.zeros(N, dtype=bool)

        # KF state x = [v(3); b_acc(3)] (bias is in world frame)
        x_state = np.zeros(6, dtype=float)
        P = np.eye(6) * 1e-2
        Q = np.zeros((6,6), dtype=float)
        Q[0:3,0:3] = np.eye(3) * VEL_PROCESS_VAR
        Q[3:6,3:6] = np.eye(3) * BIAS_PROCESS_VAR
        R_meas = np.eye(3) * MEAS_VAR_VEL

        prev_acc_lin_world = np.zeros(3)

        tprint("Starting main integration loop...")
        for i in range(1, N):
            dt = dt_array[i] if i < len(dt_array) else med_dt
            if dt <= 0:
                dt = med_dt

            # update orientation
            omega = np.array([wxs[i], wys[i], wzs[i]])  # already bias-corrected
            if USE_MADGWICK:
                try:
                    q = madgwick.updateIMU(q, gyr=omega.tolist(), acc=[axs[i], ays[i], azs[i]])
                    q = quat_normalize(q)
                except Exception:
                    dq = quat_from_omega(omega, dt)
                    q = quat_mul(q, dq); q = quat_normalize(q)
            else:
                dq = quat_from_omega(omega, dt)
                q = quat_mul(q, dq); q = quat_normalize(q)

            qs[i] = q
            R = quat_to_R(q)
            Rs[i] = R

            # transform accel to world frame, remove gravity
            acc_dev = np.array([axs[i], ays[i], azs[i]])
            acc_world = R @ acc_dev
            acc_lin_world = acc_world.copy()
            acc_lin_world[2] -= 9.81

            # stationary detection based on accel magnitude std and low gyro activity
            if i >= window_len:
                mags = np.sqrt(axs[i-window_len+1:i+1]**2 + ays[i-window_len+1:i+1]**2 + azs[i-window_len+1:i+1]**2)
                if np.std(mags) < ZUPT_STD_THRESHOLD and np.linalg.norm(omega) < 0.02:
                    stationary[i] = True

            # KF predict
            # x_pred[0:3] = v + (a_lin - b)*dt
            b = x_state[3:6]
            x_pred = np.zeros(6, dtype=float)
            x_pred[0:3] = x_state[0:3] + (acc_lin_world - b) * dt
            x_pred[3:6] = b
            # linearized F (approx)
            F = np.eye(6)
            F[0:3,3:6] = -np.eye(3) * dt
            P = F @ P @ F.T + Q

            # Measurement update if stationary: measured v = 0
            if stationary[i]:
                H = np.zeros((3,6), dtype=float)
                H[0:3,0:3] = np.eye(3)
                z = np.zeros(3)
                y = z - x_pred[0:3]
                S = H @ P @ H.T + R_meas
                K = P @ H.T @ np.linalg.inv(S)
                x_upd = x_pred + K @ y
                P = (np.eye(6) - K @ H) @ P
                x_state[:] = x_upd
            else:
                x_state[:] = x_pred

            # write back velocity and integrate position (trapezoidal)
            vel[i] = x_state[0:3]
            pos[i] = pos[i-1] + 0.5*(vel[i-1] + vel[i]) * dt

            prev_acc_lin_world = acc_lin_world

            if (i % max(1, N//10)) == 0:
                tprint(f"Processed {i}/{N} samples ({(i/N)*100:.1f}%)")

        tprint("Integration finished. Building Plotly frames...")

        # center positions for visualization
        min_vals = pos.min(axis=0)
        max_vals = pos.max(axis=0)
        span = max_vals - min_vals
        center = (max_vals + min_vals) / 2.0
        pos_centered = pos - center

        # Helper to build orientation axes traces
        def axes_traces(idx, origin):
            R = Rs[idx]
            length = max(span.mean()*0.18, 0.05)
            x_axis = origin + R[:,0]*length
            y_axis = origin + R[:,1]*length
            z_axis = origin + R[:,2]*length
            return [
                go.Scatter3d(x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
                             mode='lines', line=dict(color='red', width=5), showlegend=False),
                go.Scatter3d(x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
                             mode='lines', line=dict(color='green', width=5), showlegend=False),
                go.Scatter3d(x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
                             mode='lines', line=dict(color='blue', width=5), showlegend=False)
            ]

        frames = []
        for i in range(N):
            pts = pos_centered[:i+1]
            origin = pos_centered[i]
            data = []
            data.append(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                     mode='lines+markers', marker=dict(size=3, color='red'),
                                     line=dict(color='royalblue', width=3)))
            data.extend(axes_traces(i, origin))
            if AUTO_ROTATE:
                angle = np.deg2rad(i * ROTATE_DEG_PER_FRAME)
                cam_eye = dict(x=2.0*np.cos(angle), y=2.0*np.sin(angle), z=1.0)
            else:
                cam_eye = dict(x=1.25, y=1.25, z=1.25)
            frames.append(go.Frame(data=data, name=str(i),
                                   layout=go.Layout(scene_camera=dict(eye=cam_eye))))
            if (i % max(1, N//10)) == 0:
                tprint(f"Built {i}/{N} frames ({(i/N)*100:.1f}%)")

        # initial trace
        trace0 = go.Scatter3d(x=[pos_centered[0,0]], y=[pos_centered[0,1]], z=[pos_centered[0,2]],
                              mode='markers', marker=dict(size=4, color='red'))

        layout = go.Layout(
            title="IMU Position (corrected) + Orientation axes",
            scene=dict(xaxis=dict(title='X (m)'), yaxis=dict(title='Y (m)'), zaxis=dict(title='Z (m)'), aspectmode='auto'),
            updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate',
                            args=[None, {"frame": {"duration": int(max(1, med_dt*1000/SPEED_SCALE)), "redraw": True},
                                         "fromcurrent": True, "transition": {"duration": 0}}])])],
            sliders=[dict(steps=[dict(method='animate', args=[[f.name], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], label=str(i)) for i,f in enumerate(frames)], active=0)]
        )

        fig = go.Figure(data=[trace0], frames=frames, layout=layout)
        fig.update_layout(autosize=True)
        fig.write_html(OUT_HTML, include_plotlyjs='cdn')
        tprint(f"Wrote {OUT_HTML}")
        webbrowser.open("file://" + os.path.realpath(OUT_HTML))
        tprint("Done. Check the browser for the animation. Use Play button in the plot to start.")

    except Exception:
        tprint("ERROR during processing:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
