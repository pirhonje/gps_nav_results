#!/usr/bin/env python3
"""
CTE Analysis with 0.1m Error-Based Start Trigger.
Analysis starts only after the vehicle first gets within 10cm of the path.
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ROS2 imports
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

# --- CONFIGURATION ---
POSE_TOPIC = "/vehicle_pose"
STDEV_TOPIC = "/novatel/oem7/insstdev"
TRIM_END = 300 
START_ERROR_THRESHOLD = 0.10  # 10cm trigger

def read_route_csv(folder):
    csv_path = os.path.join(folder, "equally_spaced.csv")
    data = np.loadtxt(csv_path, delimiter=',')
    return data[:, 1], data[:, 2]

def read_bag_all(bag_path):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id="sqlite3"), 
                ConverterOptions("cdr", "cdr"))
    
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    bag_t, bag_x, bag_y = [], [], []
    q_t, q_val = [], []

    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname == POSE_TOPIC:
            msg = deserialize_message(data, get_message(type_map[POSE_TOPIC]))
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            bag_t.append(t)
            bag_x.append(msg.pose.position.x)
            bag_y.append(msg.pose.position.y)
        elif tname == STDEV_TOPIC:
            msg = deserialize_message(data, get_message(type_map[STDEV_TOPIC]))
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            uncertainty = np.sqrt(msg.latitude_stdev**2 + msg.longitude_stdev**2)
            q_t.append(t)
            q_val.append(uncertainty)

    return (np.array(bag_t), np.array(bag_x), np.array(bag_y)), (np.array(q_t), np.array(q_val))

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 script.py <folder_path>")
    
    folder = sys.argv[1]
    route_x, route_y = read_route_csv(folder)
    route_pts = np.column_stack((route_x, route_y))
    
    db3s = [f for f in os.listdir(folder) if f.endswith(".db3")]
    bag_path = os.path.join(folder, max(db3s, key=lambda x: os.path.getsize(os.path.join(folder, x))))
    
    (bag_t, bag_x, bag_y), (q_t, q_val) = read_bag_all(bag_path)

    # --- ERROR-BASED START TRIGGER ---
    start_index = 0
    trigger_found = False
    
    for i in range(len(bag_x)):
        p = np.array([bag_x[i], bag_y[i]])
        # Find distance to closest point on route for this specific bag point
        dists = np.linalg.norm(route_pts - p, axis=1)
        current_error = np.min(dists)
        
        if current_error <= START_ERROR_THRESHOLD:
            start_index = i
            trigger_found = True
            break
            
    if not trigger_found:
        print(f"Warning: Error never dropped below {START_ERROR_THRESHOLD}m. Using index 0.")

    # Apply Trim from the trigger point
    bag_t = bag_t[start_index : -TRIM_END]
    bag_x = bag_x[start_index : -TRIM_END]
    bag_y = bag_y[start_index : -TRIM_END]
    
    # --- RE-CALCULATE ERRORS FOR TRIMMED DATA ---
    closest_pts, final_errors = [], []
    for i in range(len(bag_x)):
        p = np.array([bag_x[i], bag_y[i]])
        dists = np.linalg.norm(route_pts - p, axis=1)
        idx = np.argmin(dists)
        closest_pts.append(route_pts[idx])
        final_errors.append(dists[idx])
    
    final_errors = np.array(final_errors)
    closest_pts = np.array(closest_pts)

    # Plotting Logic
    ox, oy = route_x[0], route_y[0]
    start_time = bag_t[0]

    fig = plt.figure(figsize=(14, 12))
    
    # Map
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(route_x - ox, route_y - oy, 'k--', alpha=0.3, label="Goal Path")
    lines = [[(bag_x[i]-ox, bag_y[i]-oy), (closest_pts[i,0]-ox, closest_pts[i,1]-oy)] for i in range(len(final_errors))]
    lc = LineCollection(lines, colors='red', linewidths=0.5, alpha=0.1)
    ax1.add_collection(lc)
    sc = ax1.scatter(bag_x - ox, bag_y - oy, c=final_errors, cmap='jet', s=10)
    plt.colorbar(sc, ax=ax1, label="Error [m]")
    ax1.set_aspect('equal')
    ax1.set_title(f"Analysis (Started at Error < {START_ERROR_THRESHOLD}m)")

    # Error/Quality Time Series
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(bag_t - start_time, final_errors, color='royalblue', label="Tracking Error")
    if len(q_val) > 0:
        ax2b = ax2.twinx()
        ax2b.plot(q_t - start_time, q_val, color='orange', alpha=0.6, label="GPS Uncertainty")
        ax2b.set_ylabel("GPS Uncertainty (StdDev) [m]", color='orange')
    ax2.set_ylabel("Tracking Error [m]", color='royalblue')
    ax2.grid(True, alpha=0.3)

    # Histogram
    ax3 = plt.subplot(3, 1, 3)
    ax3.hist(final_errors, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(final_errors), color='red', linestyle='--', label=f'Mean: {np.mean(final_errors):.3f}m')
    ax3.set_xlabel("Error Magnitude [m]")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()