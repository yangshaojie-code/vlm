"""Staged Task 1 pre-contact check for the pink tabletop box.

The tool is deliberately separate from the formal executor.  ``plan`` is
read-only, ``position`` moves only the base to a visual standoff, and
``approach`` optionally moves the spine then both arms to an open, bounded
pre-contact pose.  It never closes either gripper and always restores motion
commands issued by the test before exiting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import (
    FK_POSITION_TOLERANCE_M,
    LEFT_A_ROT,
    PRE_GRASP_Z0,
    RIGHT_A_ROT,
    solve_bimanual_hug_pose,
    solve_bimanual_pose,
)
from color_box_detector import detect_colored_boxes
from depth_utils import robust_depth_from_bbox
from geometry_utils import transform_point
from head_camera_kinematics import SLIDE_LIMITS
from ros2_mission_node import Ros2MissionNode


TASK_COLOR = "pink"
TASK_YAW = math.pi / 2.0
DEFAULT_STANDOFF = 0.54
DEFAULT_GRIP_HALF = 0.13
DEFAULT_TOP_TO_CENTER = 0.095
BOX_HALF_DEPTH = 0.08
TABLE_TOP_Z = 0.739
BOX_CENTER_Z = TABLE_TOP_Z + DEFAULT_TOP_TO_CENTER
DEFAULT_GRASP_FWD_OFFSET = -0.05
DEFAULT_GRASP_Z_OFFSET = 0.045
MIN_CLEARANCE = 0.02
MAX_CLEARANCE = 0.08
MAX_ARM_STEP = 0.08
MAX_SPINE_STEP = 0.10
DEFAULT_MAX_LINEAR_SPEED = 0.15
DEFAULT_MAX_ANGULAR_SPEED = 0.50
# P0 and P1 must share this path.  P0's argparse --output default used to be
# /tmp/task1_precontact_check.json, so a successful station after restart never
# updated the file P1 actually reads.
DEFAULT_POSITION_REPORT_PATH = "/workspace/baseline/outputs/task1_precontact_position_fixed.json"
MIN_NAV_LINEAR_SPEED = 0.04
MIN_NAV_ANGULAR_SPEED = 0.08
NAV_ALIGN_THRESHOLD = 0.20
FINAL_YAW_PROGRESS_EPSILON_RAD = 0.01
FINAL_YAW_STALL_TIMEOUT_SEC = 4.0
OBSERVE_BACKUP_M = 0.25
OBSERVE_BACKUP_NEAR_Y = 1.15
OBSERVE_BACKUP_TIMEOUT_SEC = 40.0
OBSERVE_BACKUP_SPEED = 0.12
OBSERVE_BACKUP_MIN_M = 0.18
MAX_OBSERVE_BACKUP_ATTEMPTS = 3


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def should_backup_to_observe(current_pose) -> bool:
    """True when the base is already near the table and the close box may be out of view."""
    current = np.asarray(current_pose, dtype=float)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError("current_pose must be a finite [x, y, yaw] vector")
    return float(current[1]) >= OBSERVE_BACKUP_NEAR_Y


def observe_backup_sufficient(traveled, requested=OBSERVE_BACKUP_M) -> bool:
    """Accept a slower reverse once the camera has likely cleared the table edge."""
    traveled = float(traveled)
    requested = float(requested)
    return traveled >= min(requested - 0.08, OBSERVE_BACKUP_MIN_M)


def navigation_command(
    current_pose,
    target_pose,
    position_tolerance: float,
    yaw_tolerance: float,
    max_linear_speed: float,
    max_angular_speed: float,
):
    """Return a bounded ``(linear, angular, phase)`` navigation command."""
    current = np.asarray(current_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if current.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite([current, target])):
        raise ValueError("current_pose and target_pose must be finite [x, y, yaw] vectors")

    delta = target[:2] - current[:2]
    distance = float(np.linalg.norm(delta))
    final_yaw_error = wrap_to_pi(target[2] - current[2])
    if distance <= position_tolerance:
        if abs(final_yaw_error) <= yaw_tolerance:
            return 0.0, 0.0, "complete"
        angular_magnitude = min(max_angular_speed, max(MIN_NAV_ANGULAR_SPEED, 1.8 * abs(final_yaw_error)))
        angular = math.copysign(float(angular_magnitude), final_yaw_error)
        return 0.0, angular, "final_yaw"

    bearing = math.atan2(delta[1], delta[0])
    heading_error = wrap_to_pi(bearing - current[2])
    angular = float(np.clip(1.8 * heading_error, -max_angular_speed, max_angular_speed))
    if abs(heading_error) > NAV_ALIGN_THRESHOLD:
        return 0.0, angular, "align_to_target"

    linear = min(float(max_linear_speed), max(MIN_NAV_LINEAR_SPEED, 0.8 * distance))
    return linear, angular, "translate"


def station_target(box_world, standoff: float = DEFAULT_STANDOFF, yaw: float = TASK_YAW) -> np.ndarray:
    """Return [x, y, yaw] with the base behind a box along its approach axis."""
    box_world = np.asarray(box_world, dtype=float)
    standoff = float(standoff)
    yaw = float(yaw)
    if box_world.shape != (3,) or not np.all(np.isfinite(box_world)):
        raise ValueError("box_world must contain three finite values")
    if not 0.40 <= standoff <= 0.70:
        raise ValueError("standoff must be within [0.40, 0.70] m")
    return np.array([
        box_world[0] - standoff * math.cos(yaw),
        box_world[1] - standoff * math.sin(yaw),
        wrap_to_pi(yaw),
    ])


def center_from_surface(surface_world, top_to_center: float = DEFAULT_TOP_TO_CENTER, yaw: float = TASK_YAW) -> np.ndarray:
    """Map the Task 1 front-face RGB-D point to the tabletop box center.

    The live fixed-layout check observed the detector's center pixel on the
    north-facing front surface, not the top surface.  The box center therefore
    lies one known half-depth north of that point; its height is the known
    tabletop surface plus the known half-height.  This is scoped to the
    non-randomized Task 1 pre-contact check, never formal execution.
    """
    surface = np.asarray(surface_world, dtype=float)
    distance = float(top_to_center)
    yaw = float(yaw)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError("surface must be a finite 3-vector")
    if not 0.05 <= distance <= 0.14:
        raise ValueError("top-to-center distance must be within [0.05, 0.14] m")
    center = surface.copy()
    center[:2] += BOX_HALF_DEPTH * np.array([math.cos(yaw), math.sin(yaw)])
    center[2] = TABLE_TOP_Z + distance
    return center


def validate_approach_geometry(box_base, clearance: float, grasp_fwd_offset: float, grasp_z_offset: float):
    """Build open-arm Cartesian targets and reject unsafe/implausible geometry."""
    box_base = np.asarray(box_base, dtype=float)
    clearance = float(clearance)
    if box_base.shape != (3,) or not np.all(np.isfinite(box_base)):
        raise ValueError("box_base must contain three finite values")
    if not MIN_CLEARANCE <= clearance <= MAX_CLEARANCE:
        raise ValueError(f"clearance must be within [{MIN_CLEARANCE}, {MAX_CLEARANCE}] m")
    if not -0.10 <= float(grasp_fwd_offset) <= 0.02:
        raise ValueError("grasp-fwd-offset must be within [-0.10, 0.02] m")
    if not 0.0 <= float(grasp_z_offset) <= 0.10:
        raise ValueError("grasp-z-offset must be within [0.0, 0.10] m")
    if not 0.30 <= box_base[0] <= 0.80 or abs(box_base[1]) > 0.18:
        raise ValueError(f"box is outside the safe base-frame approach window: {box_base.tolist()}")
    half = DEFAULT_GRIP_HALF + clearance
    center = box_base + np.array([float(grasp_fwd_offset), 0.0, float(grasp_z_offset)])
    return center + np.array([0.0, half, 0.0]), center + np.array([0.0, -half, 0.0])


def _yaw_from_odom(odom) -> float:
    q = odom.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _current_base_pose(node) -> np.ndarray:
    odom = node.sensors.odom
    if odom is None:
        raise RuntimeError("waiting for odometry")
    pose = odom.pose.pose
    return np.array([pose.position.x, pose.position.y, _yaw_from_odom(odom)], dtype=float)


def _backup_reverse(node, distance, timeout, result):
    """Reverse straight back so the head camera can see the tabletop box again."""
    distance = float(distance)
    start = _current_base_pose(node)[:2]
    deadline = time.monotonic() + float(timeout)
    traveled = 0.0
    result["observe_backup_requested_m"] = distance
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        current = _current_base_pose(node)[:2]
        traveled = float(np.linalg.norm(current - start))
        result["observe_backup_traveled_m"] = traveled
        if observe_backup_sufficient(traveled, distance):
            node.controller.stop_base()
            result["published_control_topics"] = list(dict.fromkeys(
                result["published_control_topics"] + ["/cmd_vel"]
            ))
            return traveled
        node.controller.publish_velocity(-OBSERVE_BACKUP_SPEED, 0.0)
        result["published_control_topics"] = list(dict.fromkeys(
            result["published_control_topics"] + ["/cmd_vel"]
        ))
    node.controller.stop_base()
    result["observe_backup_partial"] = True
    if observe_backup_sufficient(traveled, distance):
        return traveled
    raise TimeoutError(f"observe backup timed out; traveled={traveled:.4f} m")


def _world_to_base(node, world_point) -> np.ndarray:
    odom = node.sensors.odom
    if odom is None:
        raise RuntimeError("waiting for odometry")
    pose = odom.pose.pose
    origin = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    yaw = _yaw_from_odom(odom)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ (np.asarray(world_point) - origin)


def locate_pink(node, top_to_center: float = DEFAULT_TOP_TO_CENTER):
    snapshot = node.wait_for_snapshot(timeout_sec=4.0)
    detections = detect_colored_boxes(
        snapshot.rgb,
        TASK_COLOR,
        min_area=max(60, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 5000),
    )
    if not detections:
        raise RuntimeError("no pink box detected in the current RGB frame")
    detection = max(detections, key=lambda item: item.area * item.confidence)
    depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
    camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
    frame = snapshot.camera_frame or "head_camera"
    camera_to_world = node.transforms.lookup("odom", frame)
    surface_world = transform_point(camera_to_world, camera_point)
    center_world = center_from_surface(surface_world, top_to_center)
    center_base = _world_to_base(node, center_world)
    return {
        "bbox": list(detection.bbox),
        "pixel": list(detection.center),
        "depth_m": float(depth),
        "surface_world": surface_world.tolist(),
        "center_world": center_world.tolist(),
        "center_base": center_base.tolist(),
    }


def load_position_reference(
    path,
    node,
    position_tolerance: float,
    yaw_tolerance: float,
    *,
    allow_failed_final_yaw: bool = False,
):
    """Load a verified Task 1 station report without re-observing the close box."""
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "task1_pink_precontact_check" or report.get("stage") != "position":
        raise ValueError("position-report is not a Task 1 position-stage report")
    status = report.get("status")
    phase = report.get("navigation_phase")
    is_passed = status == "passed" and phase == "complete"
    is_final_yaw_recovery = allow_failed_final_yaw and status == "failed" and phase == "final_yaw"
    if not is_passed and not is_final_yaw_recovery:
        raise ValueError("position-report did not complete successfully or qualify for final-yaw recovery")
    if float(report.get("remaining_position_error_m", math.inf)) > float(position_tolerance):
        raise ValueError("position-report exceeds the requested position tolerance")
    if is_passed and abs(float(report.get("remaining_yaw_error_rad", math.inf))) > float(yaw_tolerance):
        raise ValueError("position-report exceeds the requested yaw tolerance")
    center_world = np.asarray(report.get("detection", {}).get("center_world"), dtype=float)
    recorded_base = report.get("final_base", report.get("last_base"))
    final_base = np.asarray(recorded_base, dtype=float)
    station = np.asarray(report.get("station_target"), dtype=float)
    if (
        center_world.shape != (3,)
        or final_base.shape != (3,)
        or station.shape != (3,)
        or not np.all(np.isfinite([center_world, final_base, station]))
    ):
        raise ValueError("position-report is missing finite target or final base coordinates")

    odom = node.sensors.odom
    if odom is None:
        raise RuntimeError("waiting for odometry")
    pose = odom.pose.pose
    current_base = np.array([pose.position.x, pose.position.y, _yaw_from_odom(odom)], dtype=float)
    drift_position = float(np.linalg.norm(current_base[:2] - final_base[:2]))
    drift_yaw = abs(wrap_to_pi(current_base[2] - final_base[2]))
    if drift_position > 0.08 or drift_yaw > 0.08:
        report_time = report.get("finished_at", "unknown")
        raise RuntimeError(
            "base moved since position report: "
            f"position drift={drift_position:.4f} m, yaw drift={drift_yaw:.4f} rad; "
            f"report_finished_at={report_time}; "
            f"report_final_base={final_base.tolist()}; current_base={current_base.tolist()}. "
            "After a Server restart, re-run P0 without --position-report and write --output to this same path."
        )
    return {
        "source": str(report_path),
        "center_world": center_world.tolist(),
        "center_base": _world_to_base(node, center_world).tolist(),
        "station_target": station.tolist(),
        "position_report_final_base": final_base.tolist(),
        "current_base": current_base.tolist(),
        "base_drift_position_m": drift_position,
        "base_drift_yaw_rad": drift_yaw,
        "recovery_from_failed_final_yaw": bool(is_final_yaw_recovery),
    }


def _current_arm_state(node):
    left_names = [f"left_arm_joint{i}" for i in range(1, 7)]
    right_names = [f"right_arm_joint{i}" for i in range(1, 7)]
    left = node.sensors.joint_vector(left_names)
    right = node.sensors.joint_vector(right_names)
    left_gripper = float(node.sensors.joint_vector(["left_arm_eef_gripper_joint"])[0])
    right_gripper = float(node.sensors.joint_vector(["right_arm_eef_gripper_joint"])[0])
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (left_gripper, right_gripper)):
        raise RuntimeError("current gripper feedback is invalid")
    return left, right, left_gripper, right_gripper


def _wait_pair(node, left_target, right_target, timeout, tolerance, stable_samples=3):
    deadline = time.monotonic() + float(timeout)
    stable = 0
    last_left = last_right = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        try:
            last_left, last_right, _lg, _rg = _current_arm_state(node)
        except Exception:
            try:
                last_left = node.sensors.joint_vector([f"left_arm_joint{i}" for i in range(1, 7)])
                last_right = node.sensors.joint_vector([f"right_arm_joint{i}" for i in range(1, 7)])
                if not np.all(np.isfinite(last_left)) or not np.all(np.isfinite(last_right)):
                    continue
            except Exception:
                continue
        error = max(
            float(np.max(np.abs(last_left - left_target))),
            float(np.max(np.abs(last_right - right_target))),
        )
        stable = stable + 1 if error <= tolerance else 0
        if stable >= stable_samples:
            return last_left, last_right
    raise TimeoutError(f"paired arms did not settle; left={last_left}, right={last_right}")


def _traverse_pair(node, left_start, right_start, left_end, right_end, max_step, timeout, tolerance, left_gripper, right_gripper, apply, result):
    left_start, right_start = np.asarray(left_start), np.asarray(right_start)
    left_end, right_end = np.asarray(left_end), np.asarray(right_end)
    count = max(1, int(math.ceil(max(
        float(np.max(np.abs(left_end - left_start))),
        float(np.max(np.abs(right_end - right_start))),
    ) / max_step)))
    result.setdefault("waypoint_counts", []).append(count)
    reached_left, reached_right = left_start, right_start
    if not apply:
        return reached_left, reached_right
    for index in range(1, count + 1):
        fraction = index / count
        left = left_start + fraction * (left_end - left_start)
        right = right_start + fraction * (right_end - right_start)
        node.controller.command_arm("l", left, left_gripper)
        node.controller.command_arm("r", right, right_gripper)
        result["published_control_topics"] = list(dict.fromkeys(
            result["published_control_topics"] + [
                "/left_arm_forward_position_controller/commands",
                "/right_arm_forward_position_controller/commands",
            ]
        ))
        reached_left, reached_right = _wait_pair(node, left, right, timeout, tolerance)
    return reached_left, reached_right


def _traverse_spine(node, start, end, max_step, timeout, tolerance, apply, result):
    """Move the slide in bounded increments and require feedback at each one."""
    start, end = float(start), float(end)
    count = max(1, int(math.ceil(abs(end - start) / max_step)))
    result["spine_waypoint_count"] = count
    if not apply:
        return start
    result["published_control_topics"] = list(dict.fromkeys(
        result["published_control_topics"] + ["/spine_forward_position_controller/commands"]
    ))
    reached = start
    for index in range(1, count + 1):
        target = start + index / count * (end - start)
        node.controller.command_spine(target)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            node.spin_once(0.05)
            reached = float(node.sensors.joint_vector(["slide_joint"])[0])
            if abs(reached - target) <= tolerance:
                break
        else:
            raise TimeoutError("slide feedback did not reach waypoint")
    return reached


def _navigate(
    node,
    target,
    position_tolerance,
    yaw_tolerance,
    timeout,
    max_distance,
    max_linear_speed,
    max_angular_speed,
    result,
):
    target = np.asarray(target, dtype=float)
    start = node.sensors.odom
    if start is None:
        raise RuntimeError("waiting for odometry")
    start_xy = np.array([start.pose.pose.position.x, start.pose.pose.position.y], dtype=float)
    start_pose = np.array([start_xy[0], start_xy[1], _yaw_from_odom(start)], dtype=float)
    result["initial_base"] = start_pose.tolist()
    result["base_path_length_m"] = 0.0
    if np.linalg.norm(target[:2] - start_xy) > max_distance:
        raise RuntimeError("requested base move exceeds max-distance safety bound")

    previous = start_xy
    path_length = 0.0
    deadline = time.monotonic() + timeout
    final_yaw_entered_at = None
    final_yaw_min_abs_error = math.inf
    final_yaw_last_progress_at = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        odom = node.sensors.odom
        if odom is None:
            continue
        pose = odom.pose.pose
        current = np.array([pose.position.x, pose.position.y])
        yaw = _yaw_from_odom(odom)
        step_distance = float(np.linalg.norm(current - previous))
        if math.isfinite(step_distance) and step_distance <= 0.25:
            path_length += step_distance
        previous = current
        current_pose = np.array([current[0], current[1], yaw], dtype=float)
        distance = float(np.linalg.norm(target[:2] - current))
        yaw_error = wrap_to_pi(target[2] - yaw)
        linear, angular, phase = navigation_command(
            current_pose,
            target,
            position_tolerance,
            yaw_tolerance,
            max_linear_speed,
            max_angular_speed,
        )
        result.update({
            "last_base": current_pose.tolist(),
            "base_path_length_m": path_length,
            "remaining_position_error_m": distance,
            "remaining_yaw_error_rad": yaw_error,
            "navigation_phase": phase,
        })
        now = time.monotonic()
        if phase == "final_yaw":
            if final_yaw_entered_at is None:
                final_yaw_entered_at = now
                final_yaw_last_progress_at = now
                result["final_yaw_diagnostics"] = {
                    "entered_at_monotonic": final_yaw_entered_at,
                    "initial_error_rad": yaw_error,
                    "min_abs_error_rad": abs(yaw_error),
                    "last_error_rad": yaw_error,
                }
            if abs(yaw_error) <= final_yaw_min_abs_error - FINAL_YAW_PROGRESS_EPSILON_RAD:
                final_yaw_min_abs_error = abs(yaw_error)
                final_yaw_last_progress_at = now
            diagnostics = result["final_yaw_diagnostics"]
            diagnostics["min_abs_error_rad"] = min(diagnostics["min_abs_error_rad"], abs(yaw_error))
            diagnostics["last_error_rad"] = yaw_error
            diagnostics["elapsed_sec"] = now - final_yaw_entered_at
            if now - final_yaw_last_progress_at >= FINAL_YAW_STALL_TIMEOUT_SEC:
                node.controller.stop_base()
                diagnostics["stalled"] = True
                raise TimeoutError(
                    "final yaw made no progress "
                    f"for {FINAL_YAW_STALL_TIMEOUT_SEC:.1f}s; error={yaw_error:.4f} rad"
                )
        if phase == "complete":
            node.controller.stop_base()
            result["final_base"] = current_pose.tolist()
            return
        node.controller.publish_velocity(linear, angular)
        if "/cmd_vel" not in result["published_control_topics"]:
            result["published_control_topics"].append("/cmd_vel")
    node.controller.stop_base()
    if result.get("last_base") is not None and result.get("final_base") is None:
        result["final_base"] = result["last_base"]
    raise TimeoutError("base did not reach the requested pre-contact station")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Staged Task 1 pink-box pre-contact check")
    parser.add_argument("--stage", choices=("plan", "position", "approach"), required=True)
    parser.add_argument("--standoff", type=float, default=DEFAULT_STANDOFF)
    parser.add_argument("--position-tolerance", type=float, default=0.03)
    parser.add_argument("--yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--clearance", type=float, default=0.03)
    parser.add_argument("--grasp-fwd-offset", type=float, default=DEFAULT_GRASP_FWD_OFFSET)
    parser.add_argument("--grasp-z-offset", type=float, default=DEFAULT_GRASP_Z_OFFSET)
    parser.add_argument("--top-to-center", type=float, default=DEFAULT_TOP_TO_CENTER)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.01)
    parser.add_argument("--settle-timeout", type=float, default=12.0)
    parser.add_argument("--nav-timeout", type=float, default=60.0)
    parser.add_argument("--max-distance", type=float, default=2.0)
    parser.add_argument("--max-linear-speed", type=float, default=DEFAULT_MAX_LINEAR_SPEED)
    parser.add_argument("--max-angular-speed", type=float, default=DEFAULT_MAX_ANGULAR_SPEED)
    parser.add_argument("--position-report", help="passed position-stage JSON used when the close box is outside camera view")
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--move-spine", action="store_true", help="approach-stage only: move spine to box height")
    parser.add_argument("--apply", action="store_true", help="required before any base/spine/arm command")
    parser.add_argument("--output", default=DEFAULT_POSITION_REPORT_PATH)
    args = parser.parse_args(argv)
    if args.stage == "plan" and args.apply:
        parser.error("plan stage is read-only and cannot use --apply")
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.05 <= args.max_linear_speed <= 0.20:
        parser.error("max-linear-speed must be within [0.05, 0.20] m/s")
    if not 0.20 <= args.max_angular_speed <= 0.60:
        parser.error("max-angular-speed must be within [0.20, 0.60] rad/s")
    result = {
        "mode": "task1_pink_precontact_check",
        "stage": args.stage,
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "gripper_motion_commanded": False,
        "published_control_topics": [],
        "standoff_m": args.standoff,
        "clearance_m": args.clearance,
    }
    node = None
    initial_left = initial_right = None
    initial_slide = None
    high_left = high_right = None
    spine_changed = False
    command_issued = False
    try:
        node = Ros2MissionNode(node_name="task1_precontact_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        if args.position_report and args.stage in ("position", "approach"):
            located = load_position_reference(
                args.position_report,
                node,
                args.position_tolerance,
                args.yaw_tolerance,
                allow_failed_final_yaw=args.stage == "position",
            )
            result["position_reference_used"] = True
        else:
            try:
                located = locate_pink(node, args.top_to_center)
            except RuntimeError as exc:
                if "no pink box detected" not in str(exc) or args.stage != "position" or not args.apply:
                    raise
                last_exc = exc
                located = None
                for _attempt in range(MAX_OBSERVE_BACKUP_ATTEMPTS):
                    current_pose = _current_base_pose(node)
                    if not should_backup_to_observe(current_pose):
                        raise last_exc
                    result.setdefault("observe_backup_from", current_pose.tolist())
                    result["observe_backup_attempts"] = int(result.get("observe_backup_attempts", 0)) + 1
                    command_issued = True
                    _backup_reverse(node, OBSERVE_BACKUP_M, OBSERVE_BACKUP_TIMEOUT_SEC, result)
                    try:
                        located = locate_pink(node, args.top_to_center)
                        result["detection_after_observe_backup"] = True
                        break
                    except RuntimeError as retry_exc:
                        if "no pink box detected" not in str(retry_exc):
                            raise
                        last_exc = retry_exc
                if located is None:
                    raise last_exc
        result["detection"] = located
        station = np.asarray(
            located.get("station_target", station_target(located["center_world"], args.standoff)),
            dtype=float,
        )
        result["station_target"] = station.tolist()
        if args.stage == "plan":
            result["status"] = "dry_run"
            return 0
        if args.stage == "position":
            if not args.apply:
                result["status"] = "dry_run"
                return 0
            command_issued = True
            _navigate(
                node, station, args.position_tolerance, args.yaw_tolerance,
                args.nav_timeout, args.max_distance,
                args.max_linear_speed, args.max_angular_speed, result,
            )
            result["status"] = "passed"
            return 0

        initial_left, initial_right, left_gripper, right_gripper = _current_arm_state(node)
        if initial_slide > 0.10:
            raise RuntimeError("slide must start at a safe high posture (<= 0.10 m) before pre-contact approach")
        current_base = np.asarray(located["center_base"], dtype=float)
        left_target, right_target = validate_approach_geometry(
            current_base, args.clearance, args.grasp_fwd_offset, args.grasp_z_offset,
        )
        target_z = float(current_base[2] + args.grasp_z_offset)
        target_slide = float(PRE_GRASP_Z0 - target_z)
        if not SLIDE_LIMITS[0] <= target_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError(f"computed approach slide {target_slide:.4f} is outside {SLIDE_LIMITS}")
        result.update({
            "initial_slide": initial_slide,
            "target_slide": target_slide,
            "left_target_position": left_target.tolist(),
            "right_target_position": right_target.tolist(),
        })
        high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
        result["high_pregrasp_plan"] = high_plan
        if abs(initial_slide - target_slide) > args.spine_tolerance and not args.move_spine:
            raise RuntimeError("slide is not at approach height; rerun approach with --move-spine --apply")
        if args.apply:
            command_issued = True
            high_left, high_right = _traverse_pair(
                node, initial_left, initial_right,
                high_plan["left_joint_target"], high_plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, 0.010,
                left_gripper, right_gripper, True, result,
            )
            result["reached_high_pregrasp_left"] = high_left.tolist()
            result["reached_high_pregrasp_right"] = high_right.tolist()
        else:
            high_left = np.asarray(high_plan["left_joint_target"])
            high_right = np.asarray(high_plan["right_joint_target"])
        if args.move_spine and abs(initial_slide - target_slide) > args.spine_tolerance:
            # Register recovery before the first waypoint: a timeout can occur
            # after the controller has accepted a partial slide motion.
            spine_changed = bool(args.apply)
            command_issued = command_issued or spine_changed
            _traverse_spine(
                node, initial_slide, target_slide, args.spine_max_step,
                args.settle_timeout, args.spine_tolerance, args.apply, result,
            )
            if args.position_report:
                located = load_position_reference(
                    args.position_report, node, args.position_tolerance, args.yaw_tolerance,
                )
            else:
                located = locate_pink(node, args.top_to_center)
            current_base = np.asarray(located["center_base"], dtype=float)
            left_target, right_target = validate_approach_geometry(current_base, args.clearance, args.grasp_fwd_offset, args.grasp_z_offset)
            result["detection_after_spine"] = located
        plan = solve_bimanual_pose(
            target_slide,
            high_left,
            high_right,
            left_target,
            right_target,
        )
        result["fk_plan"] = plan
        if not args.apply:
            result["status"] = "dry_run"
            return 0
        reached_left, reached_right = _traverse_pair(
            node, high_left, high_right,
            plan["left_joint_target"], plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, 0.010,
            left_gripper, right_gripper, True, result,
        )
        result["reached_left"] = reached_left.tolist()
        result["reached_right"] = reached_right.tolist()
        hold_until = time.monotonic() + args.hold_seconds
        while time.monotonic() < hold_until:
            node.spin_once(min(0.05, hold_until - time.monotonic()))
        _traverse_pair(
            node, reached_left, reached_right, high_left, high_right,
            args.joint_max_step, args.settle_timeout, 0.010,
            left_gripper, right_gripper, True, result,
        )
        if args.move_spine and abs(initial_slide - target_slide) > args.spine_tolerance:
            _traverse_spine(
                node, target_slide, initial_slide, args.spine_max_step,
                args.settle_timeout, args.spine_tolerance, True, result,
            )
        _traverse_pair(
            node, high_left, high_right, initial_left, initial_right,
            args.joint_max_step, args.settle_timeout, 0.010,
            left_gripper, right_gripper, True, result,
        )
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if command_issued and node is not None and args.stage == "approach" and initial_left is not None and initial_right is not None:
            try:
                current_left = node.sensors.joint_vector([f"left_arm_joint{i}" for i in range(1, 7)])
                current_right = node.sensors.joint_vector([f"right_arm_joint{i}" for i in range(1, 7)])
                left_gripper = float(node.sensors.joint_vector(["left_arm_eef_gripper_joint"])[0])
                right_gripper = float(node.sensors.joint_vector(["right_arm_eef_gripper_joint"])[0])
                if high_left is not None and high_right is not None:
                    _traverse_pair(
                        node, current_left, current_right, high_left, high_right,
                        args.joint_max_step, args.settle_timeout, 0.015,
                        left_gripper, right_gripper, True, result,
                    )
                    current_left, current_right = high_left, high_right
                if spine_changed and initial_slide is not None:
                    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
                    _traverse_spine(
                        node, current_slide, initial_slide, args.spine_max_step,
                        args.settle_timeout, args.spine_tolerance, True, result,
                    )
                    result["spine_return_after_failure_published"] = True
                _traverse_pair(
                    node, current_left, current_right,
                    initial_left, initial_right, args.joint_max_step, args.settle_timeout, 0.015,
                    left_gripper, right_gripper, True, result,
                )
            except Exception as restore_exc:
                result["restore_error"] = str(restore_exc)
        if node is not None:
            try:
                node.controller.stop_base()
            except Exception:
                pass
        return 2
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"could not write report {args.output}: {exc}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if node is not None:
            if command_issued:
                try:
                    node.controller.stop_base()
                    result["emergency_base_stop_published"] = True
                except Exception as exc:
                    result["emergency_base_stop_error"] = str(exc)
            node.close(stop_robot=False)


if __name__ == "__main__":
    raise SystemExit(main())
