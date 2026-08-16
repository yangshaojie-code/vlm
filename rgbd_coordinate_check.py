"""Read-only RGB-D/world-coordinate diagnostic for the fixed competition layout.

The node only creates subscriptions. It never creates a control publisher or
sends a robot command. Fixed-layout coordinates are calibration references,
not targets for the formal randomized mission.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from color_box_detector import detect_colored_boxes
from depth_utils import robust_depth_from_bbox
from geometry_utils import transform_point
from head_camera_kinematics import base_to_head_camera_from_joint_state
from ros_contract import (
    DEPTH_TOPIC,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    RGB_CAMERA_INFO_TOPIC,
    RGB_TOPIC,
    TF_TOPIC,
)
from ros_sensor_utils import SensorCache, TransformStore


FIXED_BOX_CENTERS = {
    "pink": np.array([-1.00, 2.20, 0.834]),
    "yellow": np.array([-0.54, 2.30, 1.004]),
    "brown": np.array([-2.63, 0.778, 0.837]),
}
BOX_HALF_EXTENTS = np.array([0.12, 0.08, 0.095])


def _ray_box_entry(camera_origin, box_center, half_extents=BOX_HALF_EXTENTS):
    """Return the first AABB surface point on the camera-to-center ray."""
    origin = np.asarray(camera_origin, dtype=float)
    center = np.asarray(box_center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    direction = center - origin
    lower, upper = center - half, center + half
    entry, exit_ = -float("inf"), float("inf")
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                raise ValueError("camera-to-center ray does not intersect the box")
            continue
        first = (lower[axis] - origin[axis]) / direction[axis]
        second = (upper[axis] - origin[axis]) / direction[axis]
        entry = max(entry, min(first, second))
        exit_ = min(exit_, max(first, second))
    if entry > exit_ or not 0.0 <= entry <= 1.0:
        raise ValueError("camera origin is not outside a visible box intersection")
    return origin + entry * direction


def _project_camera_point(point, intrinsics):
    x, y, z = np.asarray(point, dtype=float)
    if not np.all(np.isfinite((x, y, z))) or z <= 0:
        return None
    return np.array([
        intrinsics.fx * x / z + intrinsics.cx,
        intrinsics.fy * y / z + intrinsics.cy,
    ])


def _annotate(rgb, detections, expected_pixels, output):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = {"pink": "magenta", "yellow": "yellow", "brown": "orange"}
    for detection in detections:
        draw.rectangle(detection.bbox, outline=colors[detection.color], width=3)
        draw.ellipse(
            (detection.center[0] - 4, detection.center[1] - 4,
             detection.center[0] + 4, detection.center[1] + 4),
            outline="white", width=2,
        )
        draw.text((detection.bbox[0], max(0, detection.bbox[1] - 14)), detection.color, fill="white")
    for color, pixel in expected_pixels.items():
        if pixel is None:
            continue
        u, v = pixel
        draw.line((u - 7, v, u + 7, v), fill="cyan", width=2)
        draw.line((u, v - 7, u, v + 7), fill="cyan", width=2)
        draw.text((u + 9, v - 7), f"GT {color}", fill="cyan")
    image.save(output)


def _summary(rows, requested_colors):
    result = {}
    for color in requested_colors:
        selected = [row for row in rows if row["color"] == color and row["detected"]]
        entry = {
            "requested_samples": sum(1 for row in rows if row["color"] == color),
            "detected_samples": len(selected),
        }
        entry["detection_rate"] = (
            len(selected) / entry["requested_samples"] if entry["requested_samples"] else 0.0
        )
        for key in (
            "pixel_error_px", "surface_depth_error_m",
            "surface_world_error_m", "raw_world_error_m",
        ):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            if values:
                entry[f"median_{key}"] = float(np.median(values))
                entry[f"max_{key}"] = float(np.max(values))
        if selected:
            points = np.asarray([row["observed_world_surface"] for row in selected], dtype=float)
            entry["median_observed_world_surface"] = np.median(points, axis=0).tolist()
            entry["fixed_box_center"] = FIXED_BOX_CENTERS[color].tolist()
        result[color] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only fixed-layout RGB-D coordinate check")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--colors", nargs="+", choices=sorted(FIXED_BOX_CENTERS), default=list(FIXED_BOX_CENTERS))
    parser.add_argument("--output-dir", default="/tmp/rgbd_coordinate_check")
    args = parser.parse_args(argv)
    if args.samples <= 0 or args.interval < 0 or args.timeout <= 0:
        parser.error("samples and timeout must be positive; interval must be non-negative")

    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from tf2_msgs.msg import TFMessage
    except ImportError as exc:
        raise SystemExit(f"ROS 2 Python dependencies unavailable: {exc}") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensors = SensorCache()
    transforms = TransformStore()
    callback_errors = []

    def guarded(label, callback):
        def wrapped(message):
            try:
                callback(message)
            except Exception as exc:
                callback_errors.append(f"{label}: {exc}")
        return wrapped

    def update_joints(message):
        sensors.update_joint_state(message)
        matrix = base_to_head_camera_from_joint_state(message.name, message.position)
        transforms.set_transform("base_link", "head_camera", matrix)

    rclpy.init()
    node = rclpy.create_node("rgbd_coordinate_check")
    node.create_subscription(Image, RGB_TOPIC, guarded("rgb", sensors.update_rgb), qos_profile_sensor_data)
    node.create_subscription(Image, DEPTH_TOPIC, guarded("depth", sensors.update_depth), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, RGB_CAMERA_INFO_TOPIC, guarded("camera_info", sensors.update_camera_info), qos_profile_sensor_data)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, guarded("joint_state", update_joints), qos_profile_sensor_data)
    node.create_subscription(Odometry, ODOM_TOPIC, sensors.update_odom, qos_profile_sensor_data)
    node.create_subscription(TFMessage, TF_TOPIC, transforms.update, qos_profile_sensor_data)

    rows = []
    last_stamp = None
    deadline = time.monotonic() + args.timeout
    try:
        while len({row["sample"] for row in rows}) < args.samples and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                snapshot = sensors.wait_snapshot(timeout=0.0)
                camera_to_world = transforms.lookup("odom", snapshot.camera_frame or "head_camera")
            except (TimeoutError, RuntimeError):
                continue
            if sensors.odom is None or snapshot.rgb_stamp == last_stamp:
                continue
            last_stamp = snapshot.rgb_stamp
            sample_id = len({row["sample"] for row in rows}) + 1
            detections = detect_colored_boxes(snapshot.rgb, min_area=max(60, snapshot.rgb.size // 15000))
            by_color = {}
            for detection in detections:
                current = by_color.get(detection.color)
                if current is None or detection.area * detection.confidence > current.area * current.confidence:
                    by_color[detection.color] = detection

            world_to_camera = np.linalg.inv(camera_to_world)
            camera_origin_world = transform_point(camera_to_world, np.zeros(3))
            expected_pixels = {}
            expected_surfaces = {}
            for color in args.colors:
                expected_center_camera = transform_point(world_to_camera, FIXED_BOX_CENTERS[color])
                expected_surface_world = _ray_box_entry(camera_origin_world, FIXED_BOX_CENTERS[color])
                expected_surface_camera = transform_point(world_to_camera, expected_surface_world)
                expected_surfaces[color] = expected_surface_world
                expected_pixels[color] = _project_camera_point(expected_center_camera, snapshot.intrinsics)
                detection = by_color.get(color)
                row = {"sample": sample_id, "stamp": snapshot.rgb_stamp, "color": color, "detected": detection is not None}
                if detection is not None:
                    try:
                        depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
                        camera_surface = snapshot.intrinsics.project_pixel(*detection.center, depth)
                        world_surface = transform_point(camera_to_world, camera_surface)
                        expected_pixel = expected_pixels[color]
                        row.update({
                            "bbox": list(detection.bbox),
                            "detected_pixel": list(detection.center),
                            "expected_center_pixel": expected_pixel.tolist() if expected_pixel is not None else None,
                            "pixel_error_px": float(np.linalg.norm(np.asarray(detection.center) - expected_pixel)) if expected_pixel is not None else None,
                            "surface_depth_m": float(depth),
                            "expected_surface_depth_m": float(expected_surface_camera[2]) if expected_surface_camera[2] > 0 else None,
                            "surface_depth_error_m": float(abs(expected_surface_camera[2] - depth)) if expected_surface_camera[2] > 0 else None,
                            "observed_world_surface": world_surface.tolist(),
                            "expected_world_surface": expected_surface_world.tolist(),
                            "surface_world_delta": (world_surface - expected_surface_world).tolist(),
                            "surface_world_error_m": float(np.linalg.norm(world_surface - expected_surface_world)),
                            "fixed_box_center": FIXED_BOX_CENTERS[color].tolist(),
                            "world_delta_surface_to_center": (world_surface - FIXED_BOX_CENTERS[color]).tolist(),
                            "raw_world_error_m": float(np.linalg.norm(world_surface - FIXED_BOX_CENTERS[color])),
                        })
                    except Exception as exc:
                        row["detected"] = False
                        row["error"] = str(exc)
                rows.append(row)

            _annotate(snapshot.rgb, list(by_color.values()), expected_pixels, output_dir / f"sample_{sample_id:02d}.png")
            end = time.monotonic() + args.interval
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=min(0.05, end - time.monotonic()))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    collected = len({row["sample"] for row in rows})
    report = {
        "mode": "read_only_fixed_layout_coordinate_check",
        "requested_samples": args.samples,
        "collected_samples": collected,
        "colors": args.colors,
        "callback_errors": callback_errors,
        "notes": [
            "Cyan cross is the projection of the fixed-layout box center; rectangle is the RGB detector.",
            "surface_world_error_m compares RGB-D with the expected visible AABB surface and is the primary coordinate metric.",
            "raw_world_error_m compares a visible surface with the physical center and therefore includes the box half-size.",
            "Fixed coordinates must not be used as formal randomized-layout motion targets.",
        ],
        "summary": _summary(rows, args.colors),
        "samples": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={output_dir / 'report.json'}")
    print(f"images={output_dir / 'sample_*.png'}")
    if collected < args.samples:
        print(f"ERROR: collected only {collected}/{args.samples} synchronized frames")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
