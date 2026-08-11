"""Optional ROS 2 subscription adapter for the formal Server contract.

The module is importable on the host without ROS installed. In the official
Client container, instantiate ``Ros2MissionNode`` and connect its callbacks to
the motion/navigation implementation once the image's control topics are
known.
"""

import json
import os
import threading
import time
from typing import Optional

from head_camera_kinematics import base_to_head_camera_from_joint_state
from mission_orchestrator import MissionOrchestrator
from mission_protocol import MissionProtocolError
from ros_contract import (
    GAME_INFO_TOPIC,
    INSTRUCTION_TOPIC,
    SCORE_TOPIC,
    TASK_INFO_TOPIC,
    DEPTH_CAMERA_INFO_TOPIC,
    DEPTH_TOPIC,
    JOINT_STATES_TOPIC,
    LEFT_WRIST_RGB_TOPIC,
    ODOM_TOPIC,
    RGB_CAMERA_INFO_TOPIC,
    RGB_TOPIC,
    RIGHT_WRIST_RGB_TOPIC,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    parse_gameinfo_message,
    parse_instruction_message,
    parse_score_message,
)
from ros_robot_control import RosRobotController
from ros_sensor_utils import SensorCache, TransformStore


class Ros2UnavailableError(RuntimeError):
    pass


class Ros2MissionNode:
    """Subscribe to mission/referee topics and retain the latest normalized data."""

    def __init__(self, orchestrator: Optional[MissionOrchestrator] = None, node_name: str = "material_sorting_client", auto_init: bool = True):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import CameraInfo, Image, JointState
            from std_msgs.msg import Float64MultiArray, Int32, String
            from tf2_msgs.msg import TFMessage
        except ImportError as exc:
            raise Ros2UnavailableError("当前 Python 环境没有 ROS 2 rclpy/std_msgs") from exc

        self._rclpy = rclpy
        if auto_init and not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)
        self.orchestrator = orchestrator or MissionOrchestrator()
        self.latest_taskinfo = None
        self.latest_score = None
        self.latest_instruction_raw = None
        self.latest_left_rgb = None
        self.latest_right_rgb = None
        self.errors = []
        self.sensors = SensorCache()
        self.transforms = TransformStore()
        self.camera_transform_source = "unavailable"
        self._mission_condition = threading.Condition()
        self.node.create_subscription(String, INSTRUCTION_TOPIC, self._instruction_callback, 10)
        self.node.create_subscription(String, TASK_INFO_TOPIC, self._taskinfo_callback, 10)
        self.node.create_subscription(String, GAME_INFO_TOPIC, self._gameinfo_callback, 10)
        self.node.create_subscription(Int32, SCORE_TOPIC, self._score_callback, 10)
        sensor_qos = qos_profile_sensor_data
        self.node.create_subscription(Image, RGB_TOPIC, self._sensor_callback(self.sensors.update_rgb, "rgb"), sensor_qos)
        self.node.create_subscription(Image, DEPTH_TOPIC, self._sensor_callback(self.sensors.update_depth, "depth"), sensor_qos)
        self.node.create_subscription(CameraInfo, RGB_CAMERA_INFO_TOPIC, self._sensor_callback(self.sensors.update_camera_info, "camera_info"), sensor_qos)
        self.node.create_subscription(CameraInfo, DEPTH_CAMERA_INFO_TOPIC, self._sensor_callback(self.sensors.update_camera_info, "depth_camera_info"), sensor_qos)
        self.node.create_subscription(Image, LEFT_WRIST_RGB_TOPIC, self._left_rgb_callback, sensor_qos)
        self.node.create_subscription(Image, RIGHT_WRIST_RGB_TOPIC, self._right_rgb_callback, sensor_qos)
        self.node.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._sensor_callback(self._joint_state_callback, "joint_state"),
            sensor_qos,
        )
        self.node.create_subscription(Odometry, ODOM_TOPIC, self.sensors.update_odom, sensor_qos)
        self.node.create_subscription(TFMessage, TF_TOPIC, self.transforms.update, sensor_qos)
        tf_static_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(TFMessage, TF_STATIC_TOPIC, self.transforms.update, tf_static_qos)
        self.controller = RosRobotController(self.node, Twist, Float64MultiArray, self.sensors)
        self._load_static_camera_transform()

    def _instruction_callback(self, message) -> None:
        raw = getattr(message, "data", message)
        # The server publishes the same volatile instruction repeatedly. Do
        # not reload the mission after GAME_DONE or while an attempt is active;
        # reloading would clear the retry/context history and could start a
        # second physical game on the same scene.
        if self.latest_instruction_raw == raw and self.orchestrator.mission is not None:
            return
        if self.orchestrator.state.value not in ("GAME_INIT", "GAME_DONE", "TIMEOUT"):
            self.node.get_logger().warning("ignoring a different instruction while a mission is active")
            return
        try:
            self.orchestrator.load_mission(parse_instruction_message(message))
            self.orchestrator.start_game()
            self.latest_instruction_raw = raw
            with self._mission_condition:
                self._mission_condition.notify_all()
        except (MissionProtocolError, RuntimeError, ValueError) as exc:
            self.errors.append(f"instruction: {exc}")
            self.node.get_logger().error(str(exc))

    def _taskinfo_callback(self, message) -> None:
        self.latest_taskinfo = getattr(message, "data", message)

    def _gameinfo_callback(self, message) -> None:
        try:
            self.orchestrator.sync_game_info(parse_gameinfo_message(message))
        except (MissionProtocolError, RuntimeError, ValueError) as exc:
            self.errors.append(f"gameinfo: {exc}")
            self.node.get_logger().error(str(exc))

    def _score_callback(self, message) -> None:
        try:
            self.latest_score = parse_score_message(message)
        except (MissionProtocolError, RuntimeError, ValueError) as exc:
            self.errors.append(f"score: {exc}")
            self.node.get_logger().error(str(exc))

    def _sensor_callback(self, callback, label):
        def wrapped(message):
            try:
                callback(message)
            except Exception as exc:
                self.errors.append(f"{label}: {exc}")
                self.node.get_logger().warning(f"{label}: {exc}")
        return wrapped

    def _left_rgb_callback(self, message) -> None:
        self.latest_left_rgb = message

    def _right_rgb_callback(self, message) -> None:
        self.latest_right_rgb = message

    def _joint_state_callback(self, message) -> None:
        self.sensors.update_joint_state(message)
        matrix = base_to_head_camera_from_joint_state(message.name, message.position)
        frame = os.environ.get("MATERIAL_HEAD_CAMERA_FRAME", "head_camera").lstrip("/")
        self.transforms.set_transform("base_link", frame, matrix)
        self.camera_transform_source = "mjcf_joint_state"

    def _load_static_camera_transform(self) -> None:
        """Load an explicit calibration override until joint feedback arrives.

        MATERIAL_CAMERA_TO_BASE is a JSON row-major 4x4 matrix. The frame name
        is MATERIAL_HEAD_CAMERA_FRAME (default: head_camera).  Normal formal
        operation replaces this with the MJCF-derived dynamic transform on the
        first valid JointState callback.
        """
        payload = os.environ.get("MATERIAL_CAMERA_TO_BASE")
        if not payload:
            return
        try:
            matrix = json.loads(payload)
            frame = os.environ.get("MATERIAL_HEAD_CAMERA_FRAME", "head_camera")
            self.transforms.set_transform("base_link", frame, matrix)
            self.camera_transform_source = "static_override"
        except Exception as exc:
            raise ValueError(f"invalid MATERIAL_CAMERA_TO_BASE: {exc}") from exc

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def wait_for_mission(self, timeout_sec: float = 10.0):
        deadline = time.monotonic() + float(timeout_sec)
        while self.orchestrator.mission is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for /material/instruction; verify Server, ROS_DOMAIN_ID and network")
            self.spin_once(min(0.1, remaining))
        return self.orchestrator.mission

    def wait_for_robot_state(self, timeout_sec: float = 5.0) -> None:
        """Wait for odometry and the joints required by either arm.

        This preflight happens before ``start_attempt`` so missing startup
        callbacks do not consume all three competition attempts immediately.
        """
        required = {
            "slide_joint", "head_yaw_joint", "head_pitch_joint",
            *(f"left_arm_joint{i}" for i in range(1, 7)),
            "left_arm_eef_gripper_joint",
            *(f"right_arm_joint{i}" for i in range(1, 7)),
            "right_arm_eef_gripper_joint",
        }
        deadline = time.monotonic() + float(timeout_sec)
        while self.sensors.odom is None or not required.issubset(self.sensors.joint_names):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(required.difference(self.sensors.joint_names))
                raise TimeoutError(f"robot state timeout: odom={self.sensors.odom is not None}, missing_joints={missing}")
            self.spin_once(min(0.05, remaining))

    def wait_for_snapshot(self, timeout_sec: float = 3.0, max_skew: float = 0.15):
        deadline = time.monotonic() + float(timeout_sec)
        while True:
            try:
                return self.sensors.wait_snapshot(timeout=0.0, max_skew=max_skew)
            except TimeoutError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.sensors.wait_snapshot(timeout=0.0, max_skew=max_skew)
                self.spin_once(min(0.05, remaining))

    def close(self, stop_robot: bool = True) -> None:
        if stop_robot:
            try:
                self.controller.stop_all()
            except Exception as exc:
                self.node.get_logger().warning(f"stop_all: {exc}")
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
