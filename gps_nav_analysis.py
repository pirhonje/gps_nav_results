#!/usr/bin/env python3
"""
CTE + overlay plot (single argument).

Folder must contain:
  - equally_spaced.csv
  - exactly one (or more) rosbag2 sqlite file: *.db3  (largest one will be used)

Reads:  /vehicle_pose
Writes: <folder>/results.txt
        <folder>/path_vs_trajectory.png

Run:
  python3 gps_nav_analysis.py /path/to/folder
"""

import csv
import math
import os
import statistics
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

# ROS2 (run inside a ROS2 environment)
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

POSE_TOPIC = "/vehicle_pose"


# ---------------- CSV ROUTE ----------------

def _looks_numeric_start(line: str) -> bool:
    tok = line.strip().split(",")[0].strip()
    if not tok:
        return False
    try:
        float(tok)
        return True
    except Exception:
        return False


def read_route_csv(folder: str) -> List[Tuple[float, float]]:
    """
    Reads equally_spaced.csv from folder.

    Supports:
      1) Headered CSV with columns like easting/northing or x/y
      2) Headerless numeric CSV. Assumes common layout:
            [idx, easting, northing, ...]
         -> Easting=row[1], Northing=row[2]
    """
    csv_path = os.path.join(folder, "equally_spaced.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing equally_spaced.csv in: {folder}")

    with open(csv_path, "r", newline="") as f:
        first_line = f.readline()
        f.seek(0)

        # Headered
        if first_line and not _looks_numeric_start(first_line):
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            lower = {h.strip().lower(): h for h in fields}

            def pick(*names: str):
                for n in names:
                    if n in lower:
                        return lower[n]
                return None

            col_e = pick("easting", "utm_e", "utm_easting", "x", "e", "east")
            col_n = pick("northing", "utm_n", "utm_northing", "y", "n", "north")
            if col_e is None or col_n is None:
                raise ValueError(f"Cannot detect UTM columns in header: {fields}")

            route = []
            for row in reader:
                if not row:
                    continue
                route.append((float(row[col_e]), float(row[col_n])))

        # Headerless numeric
        else:
            reader = csv.reader(f)
            route = []
            for row in reader:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) < 3:
                    raise ValueError(f"Route CSV row has <3 columns: {row}")
                # Your example: idx, E, N, ...
                e = float(row[1])
                n = float(row[2])
                route.append((e, n))

    if len(route) < 2:
        raise ValueError("Route must contain at least 2 points.")
    return route


# ---------------- CTE GEOMETRY ----------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def signed_cte_to_polyline(p: Tuple[float, float], route: List[Tuple[float, float]]) -> float:
    """
    Signed cross-track error (meters) from point p=(E,N) to route polyline.
    Sign convention:
      + if point is left of segment direction (p0->p1), - if right.
    """
    px, py = p
    best_d2 = float("inf")
    best_cte = 0.0

    for i in range(len(route) - 1):
        x0, y0 = route[i]
        x1, y1 = route[i + 1]
        vx, vy = x1 - x0, y1 - y0
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            continue

        t = ((px - x0) * vx + (py - y0) * vy) / vv
        t = clamp(t, 0.0, 1.0)

        cx = x0 + t * vx
        cy = y0 + t * vy
        dx = px - cx
        dy = py - cy
        d2 = dx * dx + dy * dy

        if d2 < best_d2:
            best_d2 = d2
            # z-component of 2D cross product (v x d) determines left/right
            cross = vx * dy - vy * dx
            sign = 1.0 if cross > 0 else (-1.0 if cross < 0 else 0.0)
            best_cte = sign * math.sqrt(d2)

    return best_cte


# ---------------- ROSBAG READING ----------------

def find_rosbag_db3(folder: str) -> str:
    db3s = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".db3")]
    if not db3s:
        raise FileNotFoundError(f"No .db3 rosbag found in folder: {folder}")
    # pick largest (common when multiple exist)
    db3s.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return db3s[0]


def read_vehicle_positions_xy(bag_path: str, topic: str) -> List[Tuple[float, float]]:
    """
    Reads x,y from topic. Supports:
      - geometry_msgs/msg/PoseStamped            -> msg.pose.position.{x,y}
      - geometry_msgs/msg/PoseWithCovarianceStamped -> msg.pose.pose.position.{x,y}
    """
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise ValueError(f"Topic '{topic}' not in bag. Available: {list(type_map.keys())}")

    msg_type = get_message(type_map[topic])

    poses: List[Tuple[float, float]] = []
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname != topic:
            continue

        msg = deserialize_message(data, msg_type)

        if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
        elif hasattr(msg, "pose") and hasattr(msg.pose, "pose") and hasattr(msg.pose.pose, "position"):
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
        else:
            raise TypeError(f"Unsupported message type on {topic}: {type(msg)}")

        poses.append((x, y))

    if not poses:
        raise ValueError(f"No messages read from topic '{topic}'.")
    return poses


# ---------------- METRICS + OUTPUT ----------------

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return sorted_vals[i]
    return sorted_vals[i] * (j - k) + sorted_vals[j] * (k - i)


def write_plot(folder: str, route: List[Tuple[float, float]], poses: List[Tuple[float, float]]) -> str:
    rx = [p[0] for p in route]
    ry = [p[1] for p in route]
    px = [p[0] for p in poses]
    py = [p[1] for p in poses]

    plt.figure()
    plt.plot(rx, ry, "-", linewidth=2, label="Goal path")
    plt.plot(px, py, ".", markersize=2, label="Vehicle position")
    plt.axis("equal")
    plt.xlabel("Easting [m]")
    plt.ylabel("Northing [m]")
    plt.title("Goal path vs Vehicle trajectory")
    plt.grid(True)
    plt.legend()

    out_png = os.path.join(folder, "path_vs_trajectory.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 gps_nav_analysis.py <folder_with_db3_and_equally_spaced_csv>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    route = read_route_csv(folder)
    bag_path = find_rosbag_db3(folder)
    poses = read_vehicle_positions_xy(bag_path, POSE_TOPIC)

    ctes = [signed_cte_to_polyline(p, route) for p in poses]
    abs_ctes = [abs(e) for e in ctes]

    rms = math.sqrt(sum(e * e for e in ctes) / len(ctes))
    mean_signed = statistics.fmean(ctes)
    mean_abs = statistics.fmean(abs_ctes)
    max_abs = max(abs_ctes)

    abs_sorted = sorted(abs_ctes)
    p95 = percentile(abs_sorted, 0.95)
    p99 = percentile(abs_sorted, 0.99)

    out_txt = os.path.join(folder, "results.txt")
    with open(out_txt, "w") as f:
        f.write(f"Folder: {folder}\n")
        f.write(f"Bag: {os.path.basename(bag_path)}\n")
        f.write(f"Topic: {POSE_TOPIC}\n")
        f.write(f"Route points: {len(route)}\n")
        f.write(f"Pose samples: {len(poses)}\n\n")
        f.write(f"RMS_CTE_m: {rms:.6f}\n")
        f.write(f"Mean_Signed_CTE_m: {mean_signed:.6f}\n")
        f.write(f"Mean_Abs_CTE_m: {mean_abs:.6f}\n")
        f.write(f"P95_Abs_CTE_m: {p95:.6f}\n")
        f.write(f"P99_Abs_CTE_m: {p99:.6f}\n")
        f.write(f"Max_Abs_CTE_m: {max_abs:.6f}\n")

    out_png = write_plot(folder, route, poses)

    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_png}")
    print("\nIf the dots don't sit on the line, your frames/units don't match (e.g., map vs utm).")


if __name__ == "__main__":
    main()
