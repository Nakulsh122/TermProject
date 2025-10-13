"""
visualize_gestures_side_by_side.py

Plots N random samples per gesture in side-by-side columns, with rows = signal groups.

Usage:
    python visualize_gestures_side_by_side.py
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = "imu_dataset"   # top-level folder containing gesture subfolders
SAMPLE_RATE = 100          # used for labeling x-axis only
FEATURE_ROWS = [
    ("ax", "ay", "az"),            # acceleration channels
    ("acc_mag",),                  # acceleration magnitude
    ("gx", "gy", "gz"),            # gyro channels
    ("gyro_mag",),                 # gyro magnitude
    ("dax", "day", "daz"),         # acceleration deltas
]
# --------------------------------

def plot_samples_side_by_side(filepaths, gesture_name, features_rows, save_fig=False, out_dir="viz"):
    """
    filepaths: list of CSV paths (len = ncols)
    features_rows: list of tuples, each tuple contains column names to plot on that row
    """
    ncols = len(filepaths)
    nrows = len(features_rows)
    if ncols == 0:
        return

    fig_w = max(6, 2.5 * ncols)
    fig_h = max(3, 1.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    for c_idx, path in enumerate(filepaths):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # Ignore timestamps, just use uniform sample indices
        n = len(df)
        t = np.arange(n) / SAMPLE_RATE  # purely for x-axis labeling

        for r_idx, cols in enumerate(features_rows):
            ax = axes[r_idx][c_idx]
            plotted_any = False
            for col in cols:
                if col in df.columns:
                    ax.plot(t, df[col].values, label=col, linewidth=1)
                    plotted_any = True
            if not plotted_any:
                ax.text(0.5, 0.5, f"(no {', '.join(cols)})", ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')

            if c_idx == 0:
                row_label = ", ".join(cols)
                ax.set_ylabel(row_label, fontsize=8)
            if r_idx == 0:
                fname = os.path.basename(path)
                ax.set_title(fname, fontsize=9)

            ax.tick_params(axis='both', which='major', labelsize=7)
            if r_idx == nrows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

            if c_idx == 0 and any(col in df.columns for col in cols):
                ax.legend(fontsize=6, loc='upper right')

    plt.suptitle(f"Gesture: {gesture_name} — {ncols} samples", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fig:
        os.makedirs(out_dir, exist_ok=True)
        safe_name = gesture_name.replace(" ", "_")
        out_path = os.path.join(out_dir, f"{safe_name}_samples.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")

    plt.show()


def main():
    try:
        n_per_gesture = int(input("How many samples per gesture to visualize? (e.g. 3): ").strip())
        if n_per_gesture <= 0:
            raise ValueError()
    except Exception:
        print("Invalid number, using default 3.")
        n_per_gesture = 3

    gestures = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not gestures:
        print(f"No gesture folders found in {DATA_DIR}.")
        return

    print(f"Found gestures: {gestures}")

    for gesture in gestures:
        folder = os.path.join(DATA_DIR, gesture)
        csv_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.csv')])
        if not csv_files:
            print(f" -> No CSV files in {gesture}, skipping.")
            continue

        k = min(n_per_gesture, len(csv_files))
        selected = random.sample(csv_files, k)
        print(f"\nGesture '{gesture}': plotting {k} samples (from {len(csv_files)} available)")
        plot_samples_side_by_side(selected, gesture, FEATURE_ROWS, save_fig=False)

if __name__ == "__main__":
    main()
