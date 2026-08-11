"""Feedback-driven three-task executor for the formal ROS 2 competition.

The executor deliberately fails closed when a camera frame, TF path, or joint
feedback is unavailable. A failed attempt leaves the physical scene intact so
the orchestrator can retry from the current state.
"""

import math
import os
import time
from typing import Optional

import numpy as np

from color_box_detector import detect_colored_boxes
from depth_utils import robust_depth_from_bbox
from formal_mission_runtime import AttemptResult
from geometry_utils import transform_point
from grasp_pose import DEFAULT_TOP_DOWN_RMAT
from mission_protocol import TaskSpec
from motion_planning import IKSolveError, MockIKBackend, get_ik_backend
from task_targets import remember_task1_source, resolve_place_world, validate_task_context


class AttemptExecutionError(RuntimeError):
    pass


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosMissionExecutor:
    def __init__(self, mission_node, arm: Optional[str] = None, ik_backend=None):
        self.node = mission_node
        self.controller = mission_node.controller
        self.sensors = mission_node.sensors
        self.transforms = mission_node.transforms
        self.ik = ik_backend or get_ik_backend(prefer_real=True)
        if isinstance(self.ik, MockIKBackend):
            raise AttemptExecutionError(
                "formal_client requires the official MMK2Kdl backend; MockIKBackend is offline-only"
            )
        self.arm = arm
        self.world_frame = os.environ.get("MATERIAL_WORLD_FRAME", "odom").lstrip("/")
        self.camera_frame = os.environ.get("MATERIAL_HEAD_CAMERA_FRAME", "head_camera").lstrip("/")
        self.action_timeout = float(os.environ.get("MATERIAL_ACTION_TIMEOUT", "90"))
        self.navigate_timeout = float(os.environ.get("MATERIAL_NAV_TIMEOUT", "35"))
        self.gripper_open = float(os.environ.get("MATERIAL_GRIPPER_OPEN", "1.0"))
        self.gripper_closed = float(os.environ.get("MATERIAL_GRIPPER_CLOSED", "0.10"))
        # The current executor below is retained for offline flow tests only.
        # Formal motion stays disabled until the dual-arm hug/release strategy
        # is ported and calibrated against the fixed Server layout.
        self.motion_ready = False
        self.motion_block_reason = "dual-arm formal action strategy is not calibrated"
        self.start_pose = None
        self._attempt_deadline = None

    def preflight(self, timeout_sec: float = 10.0) -> None:
        """Validate feedback, RGB-D calibration, and TF before an attempt."""
        started = time.monotonic()
        self.node.wait_for_robot_state(timeout_sec=timeout_sec)
        remaining = timeout_sec - (time.monotonic() - started)
        if remaining <= 0:
            raise AttemptExecutionError("preflight timed out before RGB-D validation")
        snapshot = self.node.wait_for_snapshot(timeout_sec=remaining)
        frame = snapshot.camera_frame or self.camera_frame
        if not frame:
            raise AttemptExecutionError("camera frame is empty; set MATERIAL_HEAD_CAMERA_FRAME")
        self.transforms.lookup(self.world_frame, frame)
        source = getattr(self.node, "camera_transform_source", "unknown")
        self.node.node.get_logger().info(
            f"preflight IK={type(self.ik).__name__} camera_frame={frame} extrinsics={source}"
        )

    def execute_attempt(self, task: TaskSpec, context: dict, attempt: int) -> AttemptResult:
        started = time.monotonic()
        self._attempt_deadline = started + self.action_timeout
        baseline_score = self.node.latest_score
        baseline_info_score = self.node.orchestrator.last_game_info.score if self.node.orchestrator.last_game_info else None
        try:
            self._check_budget()
            if self.start_pose is None:
                self.start_pose = self._odom_pose()
            validate_task_context(task, context)
            target_world, target_base = self._locate_target(task)
            if task.task == 1:
                remember_task1_source(context, target_world, self._side_from_base(target_base))
            place_world = resolve_place_world(task, context)
            arm = self.arm or ("l" if target_base[1] >= 0 else "r")
            self._navigate_near(target_world, started)
            # Re-observe after navigation; this avoids using a stale camera frame.
            target_world, target_base = self._locate_target(task)
            pick_position = target_base
            q_pick_approach = self._solve(pick_position + [0.0, 0.0, 0.10], arm)
            q_pick = self._solve(pick_position, arm)
            self._move_arm(arm, q_pick_approach, self.gripper_open)
            self._move_arm(arm, q_pick, self.gripper_open)
            self._move_arm(arm, q_pick, self.gripper_closed)
            self._move_arm(arm, q_pick_approach, self.gripper_closed)
            self._navigate_near(place_world, started)
            place_base = self._world_to_base(place_world)
            q_place_approach = self._solve(place_base + [0.0, 0.0, 0.10], arm)
            q_place = self._solve(place_base, arm)
            self._move_arm(arm, q_place_approach, self.gripper_closed)
            self._move_arm(arm, q_place, self.gripper_closed)
            self._move_arm(arm, q_place, self.gripper_open)
            self._move_arm(arm, q_place_approach, self.gripper_open)
            self._return_to_start(started)
            success = self._wait_server_settlement(
                task.task,
                timeout=min(5.0, self._remaining_budget()),
                baseline_score=baseline_score,
                baseline_info_score=baseline_info_score,
            )
            return AttemptResult(success=success, reason=None if success else "Server did not advance after return")
        except Exception as exc:
            try:
                self.controller.stop_all()
            except Exception:
                pass
            context.setdefault("recoveries", []).append({"task": task.task, "attempt": attempt, "action": f"stop_after_failure: {exc}"})
            return AttemptResult(False, reason=str(exc))
        finally:
            self._attempt_deadline = None

    def _remaining_budget(self) -> float:
        if self._attempt_deadline is None:
            return 0.0
        return max(0.0, self._attempt_deadline - time.monotonic())

    def _check_budget(self) -> None:
        if self._remaining_budget() <= 0:
            raise AttemptExecutionError("attempt action timeout")
        info = self.node.orchestrator.last_game_info
        if info is not None and info.time_seconds is not None and info.time_seconds >= 600.0:
            raise AttemptExecutionError("Server simulation deadline reached")

    def _snapshot_target(self, task: TaskSpec):
        snapshot = self.node.wait_for_snapshot(timeout_sec=3.0)
        detections = detect_colored_boxes(snapshot.rgb, task.target_color, min_area=max(60, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 5000))
        if not detections:
            raise AttemptExecutionError(f"no {task.target_color} box detected")
        detection = max(detections, key=lambda item: item.area * item.confidence)
        depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
        camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
        frame = snapshot.camera_frame or self.camera_frame
        if not frame:
            raise AttemptExecutionError("camera frame is empty; set MATERIAL_HEAD_CAMERA_FRAME")
        return snapshot, detection, self.transforms.lookup(self.world_frame, frame), camera_point

    def _locate_target(self, task: TaskSpec):
        _snapshot, _detection, camera_to_world, camera_point = self._snapshot_target(task)
        target_world = transform_point(camera_to_world, camera_point)
        return target_world, self._world_to_base(target_world)

    def _world_to_base(self, world_point) -> np.ndarray:
        world_point = np.asarray(world_point, dtype=float)
        odom = self.sensors.odom
        if odom is None:
            raise AttemptExecutionError("waiting for odometry")
        pose = odom.pose.pose
        base_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        yaw = _yaw_from_quaternion(pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        world_to_base = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        return world_to_base @ (world_point - base_position)

    def _odom_pose(self):
        odom = self.sensors.odom
        if odom is None:
            raise AttemptExecutionError("waiting for odometry")
        pose = odom.pose.pose
        return np.array([pose.position.x, pose.position.y, _yaw_from_quaternion(pose.orientation)])

    def _side_from_base(self, point) -> str:
        return "left" if float(point[1]) >= 0 else "right"

    def _solve(self, position, arm):
        values = self.sensors.joint_vector([f"{'left' if arm == 'l' else 'right'}_arm_joint{i}" for i in range(1, 7)])
        slide = float(self.sensors.joint_vector(["slide_joint"])[0])
        try:
            result = np.asarray(self.ik.solve(np.asarray(position, dtype=float), DEFAULT_TOP_DOWN_RMAT, arm, slide, values), dtype=float)
        except ValueError as exc:
            raise IKSolveError(str(exc)) from exc
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise AttemptExecutionError("IK returned an invalid joint vector")
        return result

    def _move_arm(self, arm, joints, gripper):
        self._check_budget()
        self.controller.command_arm(arm, joints, gripper)
        names = [f"{'left' if arm == 'l' else 'right'}_arm_joint{i}" for i in range(1, 7)]
        deadline = time.monotonic() + min(12.0, self._remaining_budget())
        stable = 0
        while time.monotonic() < deadline:
            self._check_budget()
            self.node.spin_once(0.05)
            try:
                current = self.sensors.joint_vector(names)
            except Exception:
                continue
            if np.max(np.abs(current - np.asarray(joints))) <= 0.06:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
        raise AttemptExecutionError("arm joint feedback did not reach target")

    def _navigate_near(self, world_target, started):
        self._check_budget()
        target = np.asarray(world_target, dtype=float)
        deadline = min(started + self.action_timeout, time.monotonic() + self.navigate_timeout)
        while time.monotonic() < deadline:
            self._check_budget()
            pose = self._odom_pose()
            delta = target[:2] - pose[:2]
            distance = float(np.linalg.norm(delta))
            if distance <= 0.58:
                self.controller.stop_base()
                return
            desired = math.atan2(delta[1], delta[0])
            error = math.atan2(math.sin(desired - pose[2]), math.cos(desired - pose[2]))
            angular = float(np.clip(1.8 * error, -0.5, 0.5))
            linear = 0.18 if abs(error) < 0.35 else 0.0
            if linear > 0 and self._front_obstacle():
                self.controller.stop_base()
                raise AttemptExecutionError("front depth is below 0.28 m")
            self.controller.publish_velocity(linear, angular)
            self.node.spin_once(0.05)
        self.controller.stop_base()
        raise AttemptExecutionError("navigation timeout")

    def _front_obstacle(self) -> bool:
        depth = self.sensors.depth
        if depth is None or depth.ndim != 2:
            return False
        h, w = depth.shape
        region = depth[int(h * 0.42):int(h * 0.62), int(w * 0.42):int(w * 0.58)]
        valid = region[np.isfinite(region) & (region > 0.05)]
        return valid.size >= 8 and float(np.percentile(valid, 10)) < 0.28

    def _return_to_start(self, started):
        if self.start_pose is None:
            return
        self._navigate_near(np.array([self.start_pose[0], self.start_pose[1], 0.0]), started)
        self.controller.stop_base()

    def _wait_server_settlement(self, task_id: int, timeout: float, baseline_score=None, baseline_info_score=None) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.node.orchestrator.last_game_info
            if info is not None:
                if info.task is not None and info.task > task_id:
                    return True
                if task_id == 3:
                    phase = str(info.phase or "").strip().lower()
                    if phase in {"done", "success", "completed", "game_done", "finish", "finished"}:
                        return True
                    if baseline_info_score is not None and info.score is not None and info.score > baseline_info_score:
                        return True
            if baseline_score is not None and self.node.latest_score is not None and self.node.latest_score > baseline_score:
                return True
            self.node.spin_once(0.1)
        return False
