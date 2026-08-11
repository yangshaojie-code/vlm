"""Collect one read-only ROS 2 Server snapshot for formal-client calibration.

Run this inside the official Client container while the Server is publishing.
It deliberately records metadata only; image pixel payloads are not written.
"""

import argparse
import json
import pathlib
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="outputs/ros2_probe.json",
        help="JSON output path relative to the current workspace",
    )
    args = parser.parse_args()

    try:
        import rclpy
        from geometry_msgs.msg import Twist  # noqa: F401 - validates ROS geometry package
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import Int32, String
        from tf2_msgs.msg import TFMessage
    except ImportError as exc:
        raise SystemExit(f"ROS 2 Python dependencies unavailable: {exc}") from exc

    from ros_contract import (
        DEPTH_CAMERA_INFO_TOPIC,
        DEPTH_TOPIC,
        GAME_INFO_TOPIC,
        INSTRUCTION_TOPIC,
        JOINT_STATES_TOPIC,
        ODOM_TOPIC,
        RGB_CAMERA_INFO_TOPIC,
        RGB_TOPIC,
        SCORE_TOPIC,
        TASK_INFO_TOPIC,
        TF_TOPIC,
        TF_STATIC_TOPIC,
    )

    values = {}
    node = None

    def stamp(message):
        header = getattr(message, "header", None)
        value = getattr(header, "stamp", None)
        if value is None:
            return None
        return float(value.sec) + float(value.nanosec) * 1e-9

    def frame(message):
        return str(getattr(getattr(message, "header", None), "frame_id", "") or "").lstrip("/")

    def record(topic, value):
        if topic not in values:
            values[topic] = value

    def record_string(topic):
        return lambda message: record(topic, {"data": str(message.data)})

    def record_image(topic):
        def callback(message):
            record(topic, {
                "height": int(message.height), "width": int(message.width),
                "encoding": str(message.encoding), "step": int(message.step),
                "is_bigendian": bool(message.is_bigendian),
                "frame_id": frame(message), "stamp": stamp(message),
            })
        return callback

    def record_camera_info(topic):
        def callback(message):
            record(topic, {
                "height": int(message.height), "width": int(message.width),
                "frame_id": frame(message), "distortion_model": str(message.distortion_model),
                "k": [float(value) for value in message.k], "stamp": stamp(message),
            })
        return callback

    def record_joints(message):
        record(JOINT_STATES_TOPIC, {
            "names": [str(value) for value in message.name],
            "position_count": len(message.position),
            "finite_positions": all(__import__("math").isfinite(float(value)) for value in message.position),
            "stamp": stamp(message),
        })

    def record_odom(message):
        pose = message.pose.pose
        record(ODOM_TOPIC, {
            "frame_id": str(message.header.frame_id),
            "child_frame_id": str(message.child_frame_id),
            "position": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            "orientation": [float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)],
            "stamp": stamp(message),
        })

    def record_tf(message):
        entries = values.setdefault(TF_TOPIC, [])
        known = {(item["parent"], item["child"]) for item in entries}
        for item in message.transforms:
            parent = str(item.header.frame_id).lstrip("/")
            child = str(item.child_frame_id).lstrip("/")
            if not parent or not child or (parent, child) in known:
                continue
            entries.append({
                "parent": parent,
                "child": child,
                "translation": [float(item.transform.translation.x), float(item.transform.translation.y), float(item.transform.translation.z)],
                "stamp": stamp(item),
            })
            known.add((parent, child))

    rclpy.init()
    node = rclpy.create_node("material_sorting_ros2_probe")
    node.create_subscription(String, INSTRUCTION_TOPIC, record_string(INSTRUCTION_TOPIC), 10)
    node.create_subscription(String, TASK_INFO_TOPIC, record_string(TASK_INFO_TOPIC), 10)
    node.create_subscription(String, GAME_INFO_TOPIC, record_string(GAME_INFO_TOPIC), 10)
    node.create_subscription(Int32, SCORE_TOPIC, lambda message: record(SCORE_TOPIC, {"data": int(message.data)}), 10)
    for topic in (RGB_TOPIC, DEPTH_TOPIC):
        node.create_subscription(Image, topic, record_image(topic), qos_profile_sensor_data)
    for topic in (RGB_CAMERA_INFO_TOPIC, DEPTH_CAMERA_INFO_TOPIC):
        node.create_subscription(CameraInfo, topic, record_camera_info(topic), qos_profile_sensor_data)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, record_joints, qos_profile_sensor_data)
    node.create_subscription(Odometry, ODOM_TOPIC, record_odom, qos_profile_sensor_data)
    node.create_subscription(TFMessage, TF_TOPIC, record_tf, qos_profile_sensor_data)
    node.create_subscription(TFMessage, TF_STATIC_TOPIC, record_tf, qos_profile_sensor_data)

    deadline = time.monotonic() + float(args.timeout)
    required = {INSTRUCTION_TOPIC, GAME_INFO_TOPIC, RGB_TOPIC, DEPTH_TOPIC, RGB_CAMERA_INFO_TOPIC, JOINT_STATES_TOPIC, ODOM_TOPIC, TF_TOPIC}
    while time.monotonic() < deadline and not required.issubset(values):
        rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))

    values["_meta"] = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "missing_required": sorted(required.difference(values)),
        "timeout_sec": float(args.timeout),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))

    node.destroy_node()
    rclpy.shutdown()
    return 0 if not values["_meta"]["missing_required"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
