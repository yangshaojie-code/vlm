"""Task 1 bounded pink-box shelf placement check.

Reuses the verified tabletop hug/lift, then follows the official baseline
states 8-16: leave the table by at least 0.20 m, turn west, raise at a
staging pose, creep into the shelf, lower, spread, reverse out, and retract.
It never enables the formal executor or returns to the end zone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0, solve_bimanual_hug_pose, solve_bimanual_pose
from head_camera_kinematics import SLIDE_LIMITS
from motion_planning import MMK2KdlBackend
from ros2_mission_node import Ros2MissionNode
from task1_bimanual_approach_campaign import (
    _current_arm_state_unbounded,
    _traverse_grippers,
    command_gripper_value,
)
from task1_pick_lift_check import (
    APPROACH_JOINT_TOLERANCE_RAD,
    CONTACT_STABLE_SAMPLES,
    RETRACTION_JOINT_TOLERANCE_RAD,
    TASK1_APPROACH_HALF_M,
    TASK1_GRASP_FWD_OFFSET_M,
    TASK1_GRASP_Z_OFFSET_M,
    TASK1_HOLD_HALF_M,
    _approach_until_reached_or_contact,
    carry_hold_ok,
    contact_approach_geometry,
    contact_clearance_schedule,
    hug_moved_from_pregrasp,
    lift_slide_target,
)
from task1_precontact_check import (
    FINAL_YAW_PROGRESS_EPSILON_RAD,
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _traverse_pair,
    _traverse_spine,
    load_position_reference,
    navigation_command,
    wrap_to_pi,
)
from task1_transport_check import (
    CARRY_CLEARANCE_MAX_M,
    DEFAULT_SQUEEZE_SECONDS,
    MIN_SQUEEZE_HOLD_SAMPLES,
    _command_hug,
    _confirm_inward_squeeze,
    _odom_pose,
    _traverse_spine_holding,
)


INSTRUCTION_PLACE_WORLD = np.array([-2.68, 0.778, 1.156], dtype=float)
PLACE_RADIUS_M = 0.24
PLACE_YAW = math.pi
PLACE_CLEARANCE_M = 0.055
PLACE_RELEASE_SPREAD_M = 0.04
STAGING_BACK_M = 0.50
SHELF_RETREAT_M = 0.32
TABLE_LEAVE_DISTANCE_M = 0.22
TABLE_LEAVE_MIN_TRAVELED_M = 0.20
TABLE_LIFT_HEIGHT_M = 0.10
SHELF_ZONE_X_MAX = -1.70
MAX_LINE_DISTANCE_M = 0.80
MAX_STAGING_NAV_M = 2.50
MAX_HOLD_LINEAR_SPEED = 0.12
MAX_HOLD_ANGULAR_SPEED = 0.60
MIN_HOLD_ANGULAR_SPEED = 0.20
MAX_SHELF_LINEAR_SPEED = 0.06
MIN_LINE_LINEAR_SPEED = 0.04
MAX_LINE_ANGULAR_SPEED = 0.30
LINE_POSITION_TOLERANCE_M = 0.02
STAGING_POSITION_TOLERANCE_M = 0.08
STAGING_YAW_TOLERANCE_RAD = 0.10
HOLD_YAW_STALL_TIMEOUT_SEC = 12.0
DEFAULT_NAV_TIMEOUT_SEC = 180.0
DEFAULT_LINE_TIMEOUT_SEC = 30.0
DEFAULT_SHELF_TIMEOUT_SEC = 60.0
DEFAULT_YAW_TIMEOUT_SEC = 40.0
APPROACH_STALL_SEC = 4.0
# Scoring radius is 0.24 m. Its room-side edge sits in the air in front of
# the cabinet (~x=-2.44).  A 0.18 m accept stopped on the lip (box x≈-2.50)
# and the box fell.  Drive to the place stand; only lower if still on-shelf.
APPROACH_ACCEPT_RADIUS_M = 0.08
MAX_PLACE_OUTWARD_M = 0.10


def validate_place_world(place_world) -> np.ndarray:
    """Reject place targets that are not the fixed-layout Task 1 shelf cell."""
    place = np.asarray(place_world, dtype=float)
    if place.shape != (3,) or not np.all(np.isfinite(place)):
        raise ValueError("place_world must be a finite [x, y, z] vector")
    if not (-2.90 <= place[0] <= -2.40 and 0.50 <= place[1] <= 1.05 and 1.00 <= place[2] <= 1.30):
        raise ValueError(f"place_world is outside the Task 1 shelf window: {place.tolist()}")
    return place


def pose_offset(start_pose, distance: float, *, reverse: bool) -> np.ndarray:
    """Return a same-yaw pose shifted along the current heading."""
    start = np.asarray(start_pose, dtype=float)
    distance = float(distance)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("start pose must be finite [x, y, yaw]")
    if not 0.05 <= distance <= MAX_LINE_DISTANCE_M:
        raise ValueError(f"line distance must be within [0.05, {MAX_LINE_DISTANCE_M:.2f}] m")
    sign = -1.0 if reverse else 1.0
    yaw = float(start[2])
    return np.array([
        start[0] + sign * distance * math.cos(yaw),
        start[1] + sign * distance * math.sin(yaw),
        yaw,
    ])


def place_stand_from_goal(place_world, place_yaw: float, held_center_base) -> np.ndarray:
    """Base xy that puts the held box center onto place_world xy at place_yaw."""
    place = validate_place_world(place_world)
    held = np.asarray(held_center_base, dtype=float)
    yaw = wrap_to_pi(place_yaw)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_center_base must be a finite 3-vector")
    if not 0.30 <= held[0] <= 0.80 or abs(held[1]) > 0.18:
        raise ValueError(f"held box is outside the safe base-frame window: {held.tolist()}")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    held_world = np.array([
        cosine * held[0] - sine * held[1],
        sine * held[0] + cosine * held[1],
    ])
    return np.array([place[0] - held_world[0], place[1] - held_world[1]], dtype=float)


def staging_pose(place_stand_xy, place_yaw: float, staging_back: float = STAGING_BACK_M) -> np.ndarray:
    """Pose 0.50 m behind the place stand, already facing the shelf."""
    stand = np.asarray(place_stand_xy, dtype=float)
    yaw = wrap_to_pi(place_yaw)
    back = float(staging_back)
    if stand.shape != (2,) or not np.all(np.isfinite(stand)):
        raise ValueError("place stand must be a finite [x, y] vector")
    if not 0.30 <= back <= 0.80:
        raise ValueError("staging back-off must be within [0.30, 0.80] m")
    return np.array([
        stand[0] - back * math.cos(yaw),
        stand[1] - back * math.sin(yaw),
        yaw,
    ])


def held_center_world(base_pose, held_center_base) -> np.ndarray:
    """Map a locked base-frame box center into the world/odom frame."""
    base = np.asarray(base_pose, dtype=float)
    held = np.asarray(held_center_base, dtype=float)
    if base.shape != (3,) or held.shape != (3,) or not np.all(np.isfinite([base, held])):
        raise ValueError("base pose and held center must be finite")
    cosine, sine = math.cos(base[2]), math.sin(base[2])
    return np.array([
        base[0] + cosine * held[0] - sine * held[1],
        base[1] + sine * held[0] + cosine * held[1],
        held[2],
    ])


def apply_slide_keep_hold(held_center_base, old_slide: float, new_slide: float) -> np.ndarray:
    """Update the locked box z after a slide-only raise or lower."""
    held = np.asarray(held_center_base, dtype=float).copy()
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_center_base must be a finite 3-vector")
    old_slide, new_slide = float(old_slide), float(new_slide)
    if not np.isfinite(old_slide) or not np.isfinite(new_slide):
        raise ValueError("slide values must be finite")
    held[2] += old_slide - new_slide
    return held


def slide_for_held_z(current_slide: float, held_z_base: float, z_world: float) -> float:
    """Slide command that puts the locked box center at the requested world z."""
    current_slide = float(current_slide)
    held_z_base = float(held_z_base)
    z_world = float(z_world)
    if not np.all(np.isfinite([current_slide, held_z_base, z_world])):
        raise ValueError("slide and height values must be finite")
    target = current_slide + (held_z_base - z_world)
    if not SLIDE_LIMITS[0] <= target <= SLIDE_LIMITS[1]:
        raise ValueError(f"shelf slide target {target:.4f} is outside {SLIDE_LIMITS}")
    return target


def release_cartesian(left_position, right_position, spread: float = PLACE_RELEASE_SPREAD_M):
    """Open the hug laterally without changing forward reach or height."""
    left = np.asarray(left_position, dtype=float)
    right = np.asarray(right_position, dtype=float)
    spread = float(spread)
    if left.shape != (3,) or right.shape != (3,) or not np.all(np.isfinite([left, right])):
        raise ValueError("release Cartesian targets must be finite 3-vectors")
    if not 0.02 <= spread <= 0.08:
        raise ValueError("release spread must be within [0.02, 0.08] m")
    return left + np.array([0.0, spread, 0.0]), right + np.array([0.0, -spread, 0.0])


def placement_error(held_world, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Local xy/z error of the locked box versus the instruction place point."""
    held = np.asarray(held_world, dtype=float)
    place = validate_place_world(place_world)
    radius = float(place_radius)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_world must be a finite 3-vector")
    if not 0.06 <= radius <= 0.30:
        raise ValueError("place radius must be within [0.06, 0.30] m")
    xy_error = float(np.linalg.norm(held[:2] - place[:2]))
    z_error = float(abs(held[2] - place[2]))
    return {
        "xy_error_m": xy_error,
        "z_error_m": z_error,
        "within_radius": xy_error <= radius,
    }


def box_inside_place_radius(base_pose, held_center_base, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Judge the locked box xy against the instruction place cylinder."""
    world = held_center_world(base_pose, held_center_base)
    error = placement_error(world, place_world, place_radius)
    error["held_world"] = world.tolist()
    return error


def shelf_inward_ok(held_world, place_world, max_outward_m: float = MAX_PLACE_OUTWARD_M) -> dict:
    """True when the box is not still hanging off the west-facing shelf lip.

    The shelf opens toward +X.  A positive outward offset means the locked
    center is still toward the room relative to place_world.
    """
    held = np.asarray(held_world, dtype=float)
    place = validate_place_world(place_world)
    limit = float(max_outward_m)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_world must be a finite 3-vector")
    if not 0.04 <= limit <= 0.18:
        raise ValueError("max outward offset must be within [0.04, 0.18] m")
    outward = float(held[0] - place[0])
    return {
        "outward_m": outward,
        "deep_enough": outward <= limit,
    }


def load_hold_resume(path) -> dict:
    """Load a failed in-hand shelf report so placement can continue without re-grasping."""
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "task1_bimanual_hug_shelf_place_check":
        raise ValueError("resume-report is not a Task 1 shelf-place report")
    phase = report.get("phase")
    if phase not in {"staging_nav", "shelf_raise", "shelf_approach"}:
        raise ValueError(f"resume-report phase {phase!r} cannot continue into the shelf")
    if not report.get("lift_completed"):
        raise ValueError("resume-report did not finish the table hug/lift")
    hold = report.get("hold_joint_targets") or {}
    left = np.asarray(hold.get("left"), dtype=float)
    right = np.asarray(hold.get("right"), dtype=float)
    held_clearance = report.get("held_center_base_at_clearance")
    held = np.asarray(
        held_clearance if held_clearance is not None else report.get("held_center_base_after_lift"),
        dtype=float,
    )
    staging = np.asarray(report.get("staging_pose"), dtype=float)
    if left.shape != (6,) or right.shape != (6,) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("resume-report is missing a locked hug")
    if held.shape != (3,) or staging.shape != (3,) or not np.all(np.isfinite(held)) or not np.all(np.isfinite(staging)):
        raise ValueError("resume-report is missing the locked box or staging pose")
    high = report.get("high_pregrasp_plan") or {}
    skip_to_approach = bool(report.get("shelf_raise_completed") or phase == "shelf_approach")
    return {
        "source": str(report_path),
        "phase": phase,
        "hold_left": left,
        "hold_right": right,
        "held_center_base": held,
        "staging_pose": staging,
        "place_world": validate_place_world(report.get("place_world", INSTRUCTION_PLACE_WORLD)),
        "high_plan": high,
        "initial_slide": float(report.get("initial_slide", 0.0)),
        "contact_slide": float(report.get("contact_slide", 0.0)),
        "lift_slide": float(report.get("lift_slide", 0.0)),
        "table_leave_completed": bool(report.get("table_leave_completed")),
        "shelf_raise_completed": bool(report.get("shelf_raise_completed")),
        "skip_to_approach": skip_to_approach,
        "hold_joint_targets": {"left": left.tolist(), "right": right.tolist()},
    }


def held_line_command(
    current_pose,
    start_pose,
    target_pose,
    direction: int,
    position_tolerance: float,
    yaw_tolerance: float,
    min_traveled_m: float = 0.0,
    max_linear_speed: float = MAX_SHELF_LINEAR_SPEED,
    min_linear_speed: float = MIN_LINE_LINEAR_SPEED,
    max_angular_speed: float = MAX_LINE_ANGULAR_SPEED,
):
    """Straight-line hold/creep command that can require a minimum travel."""
    current = np.asarray(current_pose, dtype=float)
    start = np.asarray(start_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (reverse) or 1 (forward)")
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (current, start, target)):
        raise ValueError("all base poses must be finite [x, y, yaw]")
    heading = np.array([math.cos(start[2]), math.sin(start[2])])
    remaining = float(np.linalg.norm(target[:2] - current[:2]))
    yaw_error = wrap_to_pi(start[2] - current[2])
    traveled = float(abs(np.dot(current[:2] - start[:2], heading)))
    offset = current[:2] - start[:2]
    cross_track = float(abs(heading[0] * offset[1] - heading[1] * offset[0]))
    details = {
        "phase": "translate",
        "remaining_m": remaining,
        "traveled_m": traveled,
        "cross_track_m": cross_track,
        "yaw_error_rad": yaw_error,
    }
    yaw_ready = abs(yaw_error) <= float(yaw_tolerance)
    far_enough = traveled + 1e-9 >= float(min_traveled_m)
    goal_distance = float(np.linalg.norm(target[:2] - start[:2]))
    near_goal = traveled >= max(0.0, goal_distance - 0.01)
    at_xy = remaining <= float(position_tolerance)
    if yaw_ready and far_enough and (at_xy or near_goal):
        details["phase"] = "complete"
        return 0.0, 0.0, details
    angular = float(np.clip(1.8 * yaw_error, -max_angular_speed, max_angular_speed))
    if abs(yaw_error) > 0.10:
        details["phase"] = "align"
        return 0.0, angular, details
    linear = direction * min(float(max_linear_speed), max(float(min_linear_speed), 0.7 * remaining))
    return linear, angular, details


def _record_topic(result, topic: str) -> None:
    result["published_control_topics"] = list(dict.fromkeys(result.get("published_control_topics", []) + [topic]))


def _apply_hold_command(node, left_current, right_current, left_hold, right_hold, gripper_open, hold_keeper=None):
    """Reassert the hug, optionally refreshing the squeeze if residual shrank."""
    left_hold = np.asarray(left_hold, dtype=float)
    right_hold = np.asarray(right_hold, dtype=float)
    if hold_keeper is not None:
        left_hold, right_hold, contact = hold_keeper(left_current, right_current, left_hold, right_hold)
        _command_hug(node, left_hold, right_hold, gripper_open)
        return left_hold, right_hold, contact
    _command_hug(node, left_hold, right_hold, gripper_open)
    contact = carry_hold_ok(left_current, right_current, left_hold, right_hold)
    return left_hold, right_hold, contact


def _drive_line(
    node,
    start_pose,
    target_pose,
    direction: int,
    timeout: float,
    result,
    left_joints,
    right_joints,
    gripper_open: float,
    *,
    require_hold: bool = True,
    position_tolerance: float = LINE_POSITION_TOLERANCE_M,
    min_traveled_m: float = 0.0,
    max_linear_speed: float = MAX_SHELF_LINEAR_SPEED,
    key_prefix: str = "line",
    held_center_base=None,
    place_world=None,
    place_radius: float | None = None,
    hold_keeper=None,
):
    """Drive a heading-aligned segment while reasserting a locked arm pose."""
    deadline = time.monotonic() + float(timeout)
    final = None
    max_cross_track = 0.0
    last_progress_at = time.monotonic()
    last_traveled = -1.0
    left_joints = np.asarray(left_joints, dtype=float)
    right_joints = np.asarray(right_joints, dtype=float)
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, left_gripper, right_gripper = _current_arm_state_unbounded(node)
        samples = result.setdefault(f"{key_prefix}_raw_gripper_feedback", [])
        if len(samples) < 20:
            samples.append({"left": float(left_gripper), "right": float(right_gripper)})
        if not (0.0 <= left_gripper <= 1.0 and 0.0 <= right_gripper <= 1.0):
            result[f"{key_prefix}_gripper_endpoint_warning"] = True
        if require_hold or hold_keeper is not None:
            left_joints, right_joints, contact = _apply_hold_command(
                node, left_current, right_current, left_joints, right_joints, gripper_open, hold_keeper,
            )
            result[f"{key_prefix}_contact_feedback"] = contact
            if require_hold and not contact["holding"]:
                node.controller.stop_base()
                raise RuntimeError(f"held-box contact changed during {key_prefix}: {contact}")
        else:
            _command_hug(node, left_joints, right_joints, gripper_open)
        current = _odom_pose(node)
        linear, angular, details = held_line_command(
            current, start_pose, target_pose, direction, position_tolerance, 0.05,
            min_traveled_m=min_traveled_m, max_linear_speed=max_linear_speed,
        )
        if held_center_base is not None and place_world is not None and place_radius is not None:
            inside = box_inside_place_radius(current, held_center_base, place_world, place_radius)
            depth = shelf_inward_ok(inside["held_world"], place_world)
            result[f"{key_prefix}_estimated_place_world"] = inside["held_world"]
            result[f"{key_prefix}_xy_error_m"] = inside["xy_error_m"]
            result[f"{key_prefix}_outward_m"] = depth["outward_m"]
            # Do not stop on the scoring/accept circle.  That circle's room-side
            # edge is the shelf lip.  Only finish early once the chassis is also
            # on the place stand and the box is inward of the lip.
            if (
                inside["within_radius"]
                and depth["deep_enough"]
                and details["remaining_m"] <= max(float(position_tolerance), 0.05)
            ):
                node.controller.stop_base()
                result[f"{key_prefix}_accepted_inside_radius"] = True
                result.update({
                    f"{key_prefix}_phase": details["phase"],
                    f"{key_prefix}_remaining_m": details["remaining_m"],
                    f"{key_prefix}_traveled_m": details["traveled_m"],
                    f"{key_prefix}_cross_track_m": details["cross_track_m"],
                    f"{key_prefix}_yaw_error_rad": details["yaw_error_rad"],
                })
                return current
        max_cross_track = max(max_cross_track, details["cross_track_m"])
        final = current
        result.update({
            f"{key_prefix}_phase": details["phase"],
            f"{key_prefix}_remaining_m": details["remaining_m"],
            f"{key_prefix}_traveled_m": details["traveled_m"],
            f"{key_prefix}_cross_track_m": details["cross_track_m"],
            f"{key_prefix}_yaw_error_rad": details["yaw_error_rad"],
            f"{key_prefix}_max_cross_track_m": max_cross_track,
        })
        if details["traveled_m"] >= last_traveled + 0.01:
            last_traveled = details["traveled_m"]
            last_progress_at = time.monotonic()
        elif (
            place_world is not None
            and time.monotonic() - last_progress_at >= APPROACH_STALL_SEC
        ):
            node.controller.stop_base()
            raise TimeoutError(
                f"{key_prefix} stalled after {details['traveled_m']:.3f} m; "
                f"remaining={details['remaining_m']:.3f} m; final={current.tolist()}"
            )
        if details["phase"] == "complete":
            node.controller.stop_base()
            return final
        node.controller.publish_velocity(linear, angular)
        _record_topic(result, "/cmd_vel")
    node.controller.stop_base()
    raise TimeoutError(f"{key_prefix} timed out; final={None if final is None else final.tolist()}")


def _face_yaw_holding(
    node,
    target_yaw: float,
    timeout: float,
    result,
    left_hold,
    right_hold,
    gripper_open: float,
    *,
    yaw_tolerance: float = STAGING_YAW_TOLERANCE_RAD,
    key_prefix: str = "face_yaw",
    hold_keeper=None,
):
    """Rotate in place to a heading while keeping the locked hug."""
    deadline = time.monotonic() + float(timeout)
    last_progress_at = time.monotonic()
    min_abs_error = math.inf
    final = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, _, _ = _current_arm_state_unbounded(node)
        left_hold, right_hold, contact = _apply_hold_command(
            node, left_current, right_current, left_hold, right_hold, gripper_open, hold_keeper,
        )
        result[f"{key_prefix}_contact_feedback"] = contact
        if not contact["holding"]:
            node.controller.stop_base()
            raise RuntimeError(f"held-box contact changed during {key_prefix}: {contact}")
        current = _odom_pose(node)
        yaw_error = wrap_to_pi(float(target_yaw) - current[2])
        final = current
        result.update({
            f"{key_prefix}_yaw_error_rad": yaw_error,
            f"{key_prefix}_base": current.tolist(),
        })
        if abs(yaw_error) <= float(yaw_tolerance):
            node.controller.stop_base()
            result[f"{key_prefix}_completed"] = True
            return current
        if abs(yaw_error) <= min_abs_error - FINAL_YAW_PROGRESS_EPSILON_RAD:
            min_abs_error = abs(yaw_error)
            last_progress_at = time.monotonic()
        if time.monotonic() - last_progress_at >= HOLD_YAW_STALL_TIMEOUT_SEC:
            node.controller.stop_base()
            raise TimeoutError(f"{key_prefix} stalled; yaw error={yaw_error:.4f} rad")
        angular_magnitude = min(
            MAX_HOLD_ANGULAR_SPEED,
            max(MIN_HOLD_ANGULAR_SPEED, 1.8 * abs(yaw_error)),
        )
        node.controller.publish_velocity(0.0, math.copysign(angular_magnitude, yaw_error))
        _record_topic(result, "/cmd_vel")
    node.controller.stop_base()
    raise TimeoutError(
        f"{key_prefix} timed out; final={None if final is None else final.tolist()}"
    )


def _navigate_holding(
    node,
    target,
    timeout: float,
    max_distance: float,
    result,
    left_hold,
    right_hold,
    gripper_open: float,
    *,
    position_tolerance: float = STAGING_POSITION_TOLERANCE_M,
    yaw_tolerance: float = STAGING_YAW_TOLERANCE_RAD,
    max_linear_speed: float = MAX_HOLD_LINEAR_SPEED,
    max_angular_speed: float = MAX_HOLD_ANGULAR_SPEED,
    hold_keeper=None,
):
    """Navigate to a world pose while continuously reasserting the hug."""
    target = np.asarray(target, dtype=float)
    start = _odom_pose(node)
    nav = {
        "initial_base": start.tolist(),
        "target": target.tolist(),
        "base_path_length_m": 0.0,
    }
    if float(np.linalg.norm(target[:2] - start[:2])) > float(max_distance):
        raise RuntimeError("requested held-box navigation exceeds max-distance safety bound")
    previous = start[:2].copy()
    path_length = 0.0
    deadline = time.monotonic() + float(timeout)
    final_yaw_entered_at = None
    final_yaw_min_abs_error = math.inf
    final_yaw_last_progress_at = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, _, _ = _current_arm_state_unbounded(node)
        left_hold, right_hold, contact = _apply_hold_command(
            node, left_current, right_current, left_hold, right_hold, gripper_open, hold_keeper,
        )
        nav["contact_feedback"] = contact
        if not contact["holding"]:
            node.controller.stop_base()
            raise RuntimeError(f"held-box contact changed during staging navigation: {contact}")
        current = _odom_pose(node)
        step_distance = float(np.linalg.norm(current[:2] - previous))
        if math.isfinite(step_distance) and step_distance <= 0.25:
            path_length += step_distance
        previous = current[:2].copy()
        linear, angular, phase = navigation_command(
            current, target, position_tolerance, yaw_tolerance, max_linear_speed, max_angular_speed,
        )
        yaw_error = wrap_to_pi(target[2] - current[2])
        nav.update({
            "last_base": current.tolist(),
            "base_path_length_m": path_length,
            "remaining_position_error_m": float(np.linalg.norm(target[:2] - current[:2])),
            "remaining_yaw_error_rad": yaw_error,
            "navigation_phase": phase,
        })
        now = time.monotonic()
        if phase == "final_yaw":
            if final_yaw_entered_at is None:
                final_yaw_entered_at = now
                final_yaw_last_progress_at = now
                nav["final_yaw_diagnostics"] = {
                    "entered_at_monotonic": final_yaw_entered_at,
                    "initial_error_rad": yaw_error,
                    "min_abs_error_rad": abs(yaw_error),
                    "last_error_rad": yaw_error,
                }
            if abs(yaw_error) <= final_yaw_min_abs_error - FINAL_YAW_PROGRESS_EPSILON_RAD:
                final_yaw_min_abs_error = abs(yaw_error)
                final_yaw_last_progress_at = now
            diagnostics = nav["final_yaw_diagnostics"]
            diagnostics["min_abs_error_rad"] = min(diagnostics["min_abs_error_rad"], abs(yaw_error))
            diagnostics["last_error_rad"] = yaw_error
            diagnostics["elapsed_sec"] = now - final_yaw_entered_at
            if now - final_yaw_last_progress_at >= HOLD_YAW_STALL_TIMEOUT_SEC:
                node.controller.stop_base()
                diagnostics["stalled"] = True
                raise TimeoutError(
                    "staging final yaw made no progress "
                    f"for {HOLD_YAW_STALL_TIMEOUT_SEC:.1f}s; error={yaw_error:.4f} rad"
                )
        if phase == "complete":
            node.controller.stop_base()
            nav["final_base"] = current.tolist()
            result["staging_navigation"] = nav
            return current
        node.controller.publish_velocity(linear, angular)
        _record_topic(result, "/cmd_vel")
    node.controller.stop_base()
    result["staging_navigation"] = nav
    raise TimeoutError("held-box staging navigation timed out")


def _sleep_holding(node, seconds, left_hold, right_hold, gripper_open, *, require_hold: bool = True, hold_keeper=None):
    deadline = time.monotonic() + float(seconds)
    last = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, _, _ = _current_arm_state_unbounded(node)
        left_hold, right_hold, last = _apply_hold_command(
            node, left_current, right_current, left_hold, right_hold, gripper_open, hold_keeper,
        )
        if require_hold and not last["holding"]:
            raise RuntimeError(f"held-box contact changed while settling: {last}")
    return last


def _establish_table_hold(node, args, located, result):
    """Reproduce the verified P2 tabletop hug and 0.10 m lift."""
    initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
    initial_left_gripper = command_gripper_value(raw_left_gripper)
    initial_right_gripper = command_gripper_value(raw_right_gripper)
    initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    if initial_slide > 0.10:
        raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
    box_base = np.asarray(located["center_base"], dtype=float)
    contact_slide = float(PRE_GRASP_Z0 - (box_base[2] + args.grasp_z_offset))
    if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
        raise RuntimeError("contact slide target is outside limits")
    lift_slide = lift_slide_target(contact_slide, args.lift_height)
    high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
    plans = []
    left_reference = np.asarray(high_plan["left_joint_target"])
    right_reference = np.asarray(high_plan["right_joint_target"])
    clearances = contact_clearance_schedule(args.initial_clearance, args.contact_step)
    for clearance in clearances:
        half = args.hold_half if clearance == 0.0 else args.approach_half
        left_target, right_target = contact_approach_geometry(
            box_base, clearance, args.grasp_fwd_offset, args.grasp_z_offset, half,
        )
        plan = solve_bimanual_pose(contact_slide, left_reference, right_reference, left_target, right_target)
        plan["clearance_m"] = clearance
        plan["half_gap_m"] = half
        plans.append(plan)
        left_reference, right_reference = np.asarray(plan["left_joint_target"]), np.asarray(plan["right_joint_target"])
    result.update({
        "initial_slide": initial_slide,
        "initial_raw_gripper_feedback": {"left": raw_left_gripper, "right": raw_right_gripper},
        "contact_slide": contact_slide,
        "lift_slide": lift_slide,
        "high_pregrasp_plan": high_plan,
        "contact_plans": plans,
        "contact_clearance_schedule_m": clearances,
    })
    context = {
        "initial_left": initial_left,
        "initial_right": initial_right,
        "initial_left_gripper": initial_left_gripper,
        "initial_right_gripper": initial_right_gripper,
        "initial_slide": initial_slide,
        "contact_slide": contact_slide,
        "lift_slide": lift_slide,
        "high_plan": high_plan,
        "plans": plans,
        "high_left": None,
        "high_right": None,
        "hold_left": None,
        "hold_right": None,
        "contact_plan_index": None,
        "held_center_base": box_base.copy(),
        "reached_left": None,
        "reached_right": None,
    }
    if not args.apply:
        return context

    high_left, high_right = _traverse_pair(
        node, initial_left, initial_right, high_plan["left_joint_target"], high_plan["right_joint_target"],
        args.joint_max_step, args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD,
        initial_left_gripper, initial_right_gripper, True, result,
    )
    context["high_left"], context["high_right"] = high_left, high_right
    _traverse_grippers(
        node, high_left, high_right, initial_left_gripper, initial_right_gripper,
        args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result,
    )
    _traverse_spine(node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
    reached_left, reached_right = high_left, high_right
    reached = []
    contact_plan = None
    contact_feedback = None
    hold_left = hold_right = None
    contact_plan_index = None
    for plan_index, plan in enumerate(plans):
        reached_left, reached_right, waypoint_result = _approach_until_reached_or_contact(
            node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
            args.joint_max_step, args.gripper_open, args.gripper_open,
            args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD, result,
            allow_early_contact=plan["clearance_m"] <= CARRY_CLEARANCE_MAX_M,
        )
        reached.append({
            "clearance_m": plan["clearance_m"],
            "left": np.asarray(reached_left, dtype=float).tolist(),
            "right": np.asarray(reached_right, dtype=float).tolist(),
            "contact_detected": bool(waypoint_result["contact_detected"]),
        })
        if waypoint_result["contact_detected"]:
            if plan["clearance_m"] > CARRY_CLEARANCE_MAX_M:
                result.setdefault("rejected_false_contacts", []).append({
                    "clearance_m": plan["clearance_m"], "reason": "too_open_to_carry",
                })
                continue
            if not hug_moved_from_pregrasp(reached_left, reached_right, high_left, high_right):
                result.setdefault("rejected_false_contacts", []).append({
                    "clearance_m": plan["clearance_m"], "reason": "still_at_pregrasp",
                })
                continue
            hold_left = np.asarray(plan["left_joint_target"], dtype=float)
            hold_right = np.asarray(plan["right_joint_target"], dtype=float)
            try:
                squeeze = _confirm_inward_squeeze(
                    node, hold_left, hold_right, args.gripper_open, args.squeeze_seconds, result,
                )
            except RuntimeError as exc:
                result.setdefault("rejected_false_contacts", []).append({
                    "clearance_m": plan["clearance_m"], "error": str(exc),
                    "feedback": result.get("squeeze_feedback"),
                })
                reached_left, reached_right, _, _ = _current_arm_state_unbounded(node)
                continue
            contact_plan = plan
            contact_plan_index = plan_index
            contact_feedback = waypoint_result
            result["squeeze_confirmed"] = True
            result["squeeze_feedback"] = squeeze
            break
    if contact_plan is None:
        raise TimeoutError("dual-arm contact was not established at any validated clearance waypoint")
    result["contact_detected"] = True
    result["contact_feedback"] = contact_feedback
    result["contact_clearance_detected_m"] = contact_plan["clearance_m"]
    result["reached_contact_plans"] = reached
    result["first_touch_joint_targets"] = {"left": reached_left.tolist(), "right": reached_right.tolist()}
    result["hold_joint_targets"] = {"left": hold_left.tolist(), "right": hold_right.tolist()}
    _traverse_spine_holding(
        node, contact_slide, lift_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        hold_left, hold_right, args.gripper_open, result,
    )
    result["lift_completed"] = True
    context.update({
        "high_left": high_left,
        "high_right": high_right,
        "hold_left": hold_left,
        "hold_right": hold_right,
        "contact_plan_index": contact_plan_index,
        "reached_left": reached_left,
        "reached_right": reached_right,
        "held_center_base": apply_slide_keep_hold(box_base, contact_slide, lift_slide),
    })
    return context


def _retract_arms(node, context, args, result):
    current_left, current_right, left_gripper, right_gripper = _current_arm_state_unbounded(node)
    left_gripper = float(np.clip(left_gripper, 0.0, 1.0))
    right_gripper = float(np.clip(right_gripper, 0.0, 1.0))
    high_left, high_right = context.get("high_left"), context.get("high_right")
    if high_left is not None and high_right is not None:
        _traverse_pair(
            node, current_left, current_right, high_left, high_right,
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            left_gripper, right_gripper, True, result,
        )
        current_left, current_right = high_left, high_right
    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    _traverse_spine(
        node, current_slide, context["initial_slide"], args.spine_max_step,
        args.settle_timeout, args.spine_tolerance, True, result,
    )
    home_left, home_right = _traverse_pair(
        node, current_left, current_right, context["initial_left"], context["initial_right"],
        args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
        args.gripper_open, args.gripper_open, True, result,
    )
    _traverse_grippers(
        node, home_left, home_right, args.gripper_open, args.gripper_open,
        context["initial_left_gripper"], context["initial_right_gripper"],
        args.gripper_max_step, args.settle_timeout, 0.010, result,
    )
    result["arms_retracted"] = True


def _recover(node, context, args, result, start_base, place_yaw: float, phase: str, released: bool, place_world):
    try:
        node.controller.stop_base()
    except Exception:
        pass
    holding = context.get("hold_left") is not None and context.get("hold_right") is not None and not released
    current = None
    try:
        current = _odom_pose(node)
    except Exception:
        current = None
    near_shelf = current is not None and float(current[0]) <= SHELF_ZONE_X_MAX
    if holding and current is not None and context.get("held_center_base") is not None:
        try:
            inside = box_inside_place_radius(
                current, context["held_center_base"], place_world, args.place_radius,
            )
            result["recovery_estimated_place_world"] = inside["held_world"]
            result["recovery_place_xy_error_m"] = inside["xy_error_m"]
        except Exception as exc:
            result["recovery_place_estimate_error"] = str(exc)
    if holding and near_shelf:
        result["recovery_kept_hold"] = True
        result["recovery_skipped_pull_out"] = True
        try:
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
        except Exception as exc:
            result["recovery_keep_hold_error"] = str(exc)
    elif holding and phase in {"hug_lift", "table_leave"} and start_base is not None:
        try:
            if current is not None:
                _drive_line(
                    node, current, start_base, 1, args.table_leave_timeout, result,
                    context["hold_left"], context["hold_right"], args.gripper_open,
                    require_hold=True, max_linear_speed=0.08, key_prefix="recovery_return",
                )
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            _traverse_spine_holding(
                node, current_slide, context["contact_slide"], args.spine_max_step,
                args.settle_timeout, args.spine_tolerance,
                context["hold_left"], context["hold_right"], args.gripper_open, result,
            )
            result["recovery_lowered_before_retract"] = True
        except Exception as exc:
            result["recovery_table_error"] = str(exc)
    elif released and near_shelf:
        try:
            _drive_line(
                node, current, pose_offset(current, SHELF_RETREAT_M, reverse=True), -1,
                args.retreat_timeout, result, context.get("release_left", context["hold_left"]),
                context.get("release_right", context["hold_right"]), args.gripper_open,
                require_hold=False, min_traveled_m=0.20, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                key_prefix="recovery_released_retreat",
            )
            result["recovery_released_retreat_completed"] = True
        except Exception as exc:
            result["recovery_released_retreat_error"] = str(exc)
    can_retract = (
        context.get("initial_left") is not None
        and (released or context.get("hold_left") is None or result.get("recovery_lowered_before_retract"))
    )
    if can_retract:
        try:
            _retract_arms(node, context, args, result)
        except Exception as exc:
            result["recovery_error"] = str(exc)
    elif holding:
        result["recovery_kept_hold"] = True
        try:
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
        except Exception as exc:
            result["recovery_keep_hold_error"] = str(exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 1 hug-to-shelf placement check")
    parser.add_argument("--position-report", help="passed or near-complete table station report; not used with --resume-report")
    parser.add_argument("--resume-report", help="failed in-hand shelf report used to continue from staging without re-grasping")
    parser.add_argument("--position-tolerance", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--initial-clearance", type=float, default=0.02)
    parser.add_argument("--contact-step", type=float, default=0.01)
    parser.add_argument("--grasp-fwd-offset", type=float, default=TASK1_GRASP_FWD_OFFSET_M)
    parser.add_argument("--grasp-z-offset", type=float, default=TASK1_GRASP_Z_OFFSET_M)
    parser.add_argument("--approach-half", type=float, default=TASK1_APPROACH_HALF_M)
    parser.add_argument("--hold-half", type=float, default=TASK1_HOLD_HALF_M)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-max-step", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=TABLE_LIFT_HEIGHT_M)
    parser.add_argument("--table-leave-distance", type=float, default=TABLE_LEAVE_DISTANCE_M)
    parser.add_argument("--place-world", nargs=3, type=float, default=INSTRUCTION_PLACE_WORLD.tolist())
    parser.add_argument("--place-radius", type=float, default=PLACE_RADIUS_M)
    parser.add_argument("--approach-accept-radius", type=float, default=APPROACH_ACCEPT_RADIUS_M)
    parser.add_argument("--place-yaw", type=float, default=PLACE_YAW)
    parser.add_argument("--place-clearance", type=float, default=PLACE_CLEARANCE_M)
    parser.add_argument("--release-spread", type=float, default=PLACE_RELEASE_SPREAD_M)
    parser.add_argument("--staging-back", type=float, default=STAGING_BACK_M)
    parser.add_argument("--shelf-retreat", type=float, default=SHELF_RETREAT_M)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--squeeze-seconds", type=float, default=DEFAULT_SQUEEZE_SECONDS)
    parser.add_argument("--nav-timeout", type=float, default=DEFAULT_NAV_TIMEOUT_SEC)
    parser.add_argument("--yaw-timeout", type=float, default=DEFAULT_YAW_TIMEOUT_SEC)
    parser.add_argument("--table-leave-timeout", type=float, default=DEFAULT_LINE_TIMEOUT_SEC)
    parser.add_argument("--shelf-timeout", type=float, default=DEFAULT_SHELF_TIMEOUT_SEC)
    parser.add_argument("--retreat-timeout", type=float, default=DEFAULT_LINE_TIMEOUT_SEC)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task1_shelf_place_check.json")
    args = parser.parse_args(argv)
    if args.resume_report:
        if args.position_report:
            parser.error("use either --resume-report or --position-report, not both")
    elif not args.position_report:
        parser.error("either --position-report or --resume-report is required")
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.0 <= args.gripper_open <= 1.0:
        parser.error("gripper-open is invalid")
    if args.settle_timeout <= 0 or args.nav_timeout <= 0 or args.yaw_timeout <= 0 or args.squeeze_seconds <= 0:
        parser.error("timeout arguments are invalid")
    if not TASK1_HOLD_HALF_M <= args.hold_half <= args.approach_half <= TASK1_APPROACH_HALF_M:
        parser.error("hold-half must be within [0.115, approach-half], and approach-half <= 0.13 m")
    if not 0.06 <= args.approach_accept_radius <= args.place_radius:
        parser.error("approach-accept-radius must be within [0.06, place-radius]")
    place_world = validate_place_world(args.place_world)
    place_yaw = wrap_to_pi(args.place_yaw)

    result = {
        "mode": "task1_bimanual_hug_shelf_place_check",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "box_contact_commanded": bool(args.apply),
        "base_motion_commanded": bool(args.apply),
        "transport_or_place_commanded": bool(args.apply),
        "gripper_open_target": args.gripper_open,
        "approach_half_m": args.approach_half,
        "hold_half_m": args.hold_half,
        "grasp_fwd_offset_m": args.grasp_fwd_offset,
        "grasp_z_offset_m": args.grasp_z_offset,
        "lift_height_m": args.lift_height,
        "table_leave_distance_m": args.table_leave_distance,
        "table_leave_min_traveled_m": TABLE_LEAVE_MIN_TRAVELED_M,
        "place_world": place_world.tolist(),
        "place_radius_m": args.place_radius,
        "approach_accept_radius_m": args.approach_accept_radius,
        "place_yaw_rad": place_yaw,
        "place_clearance_m": args.place_clearance,
        "release_spread_m": args.release_spread,
        "staging_back_m": args.staging_back,
        "shelf_retreat_m": args.shelf_retreat,
        "published_control_topics": [],
        "phase": "init",
    }
    node = None
    context = {
        "initial_left": None,
        "hold_left": None,
        "hold_right": None,
        "release_left": None,
        "release_right": None,
    }
    start_base = None
    phase = "init"
    released = False
    command_issued = False
    skip_to_approach = False
    try:
        node = Ros2MissionNode(node_name="task1_shelf_place_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        if args.resume_report:
            resume = load_hold_resume(args.resume_report)
            place_world = resume["place_world"]
            result["place_world"] = place_world.tolist()
            result["resumed"] = True
            result["resume_from"] = resume["source"]
            high = resume["high_plan"]
            high_left = np.asarray(high.get("left_joint_target"), dtype=float) if high.get("left_joint_target") is not None else None
            high_right = np.asarray(high.get("right_joint_target"), dtype=float) if high.get("right_joint_target") is not None else None
            context = {
                "initial_left": np.zeros(6),
                "initial_right": np.zeros(6),
                "initial_left_gripper": args.gripper_open,
                "initial_right_gripper": args.gripper_open,
                "initial_slide": resume["initial_slide"],
                "contact_slide": resume["contact_slide"],
                "high_left": None if high_left is None or high_left.shape != (6,) else high_left,
                "high_right": None if high_right is None or high_right.shape != (6,) else high_right,
                "high_plan": high,
                "hold_left": resume["hold_left"],
                "hold_right": resume["hold_right"],
                "held_center_base": resume["held_center_base"],
            }
            start_base = _odom_pose(node)
            stage_pose = resume["staging_pose"]
            place_stand = place_stand_from_goal(place_world, place_yaw, resume["held_center_base"])
            result.update({
                "initial_base": start_base.tolist(),
                "place_stand_xy": place_stand.tolist(),
                "staging_pose": stage_pose.tolist(),
                "held_center_base_after_lift": resume["held_center_base"].tolist(),
                "hold_joint_targets": resume["hold_joint_targets"],
                "lift_completed": True,
                "table_leave_completed": resume["table_leave_completed"],
                "contact_slide": resume["contact_slide"],
                "lift_slide": resume["lift_slide"],
                "initial_slide": resume["initial_slide"],
            })
            if not args.apply:
                result["status"] = "dry_run"
                result["box_contact_commanded"] = False
                result["base_motion_commanded"] = False
                result["transport_or_place_commanded"] = False
                return 0
            command_issued = True
            _sleep_holding(node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open)
            skip_to_approach = bool(resume["skip_to_approach"])
            if resume["shelf_raise_completed"]:
                result["shelf_raise_completed"] = True
                result["held_center_base_at_clearance"] = resume["held_center_base"].tolist()
        else:
            located = load_position_reference(
                args.position_report,
                node,
                args.position_tolerance,
                args.yaw_tolerance,
                allow_failed_final_yaw=True,
            )
            result["position_reference"] = located
            held_center_base = np.asarray(located["center_base"], dtype=float)
            start_base = _odom_pose(node)
            place_stand = place_stand_from_goal(place_world, place_yaw, held_center_base)
            stage_pose = staging_pose(place_stand, place_yaw, args.staging_back)
            table_leave_target = pose_offset(start_base, args.table_leave_distance, reverse=True)
            result.update({
                "initial_base": start_base.tolist(),
                "table_leave_target": table_leave_target.tolist(),
                "place_stand_xy": place_stand.tolist(),
                "staging_pose": stage_pose.tolist(),
                "held_center_base_at_grasp": held_center_base.tolist(),
            })
            phase = "hug_lift"
            result["phase"] = phase
            if not args.apply:
                contact_slide = float(PRE_GRASP_Z0 - (held_center_base[2] + args.grasp_z_offset))
                lift_slide = lift_slide_target(contact_slide, args.lift_height)
                held_after_lift = apply_slide_keep_hold(held_center_base, contact_slide, lift_slide)
                clearance_slide = slide_for_held_z(lift_slide, held_after_lift[2], place_world[2] + args.place_clearance)
                place_slide = slide_for_held_z(clearance_slide, apply_slide_keep_hold(held_after_lift, lift_slide, clearance_slide)[2], place_world[2])
                result.update({
                    "contact_slide": contact_slide,
                    "lift_slide": lift_slide,
                    "place_clearance_slide": clearance_slide,
                    "place_slide": place_slide,
                    "status": "dry_run",
                })
                result["box_contact_commanded"] = False
                result["base_motion_commanded"] = False
                result["transport_or_place_commanded"] = False
                return 0

            command_issued = True
            context = _establish_table_hold(node, args, located, result)
            start_base = _odom_pose(node)
            table_leave_target = pose_offset(start_base, args.table_leave_distance, reverse=True)
            result["table_leave_target"] = table_leave_target.tolist()
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()

            phase = "table_leave"
            result["phase"] = phase
            leave_final = _drive_line(
                node, start_base, table_leave_target, -1, args.table_leave_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, min_traveled_m=TABLE_LEAVE_MIN_TRAVELED_M,
                max_linear_speed=0.08, key_prefix="table_leave",
            )
            result["table_leave_final_base"] = leave_final.tolist()
            result["table_leave_completed"] = True
            if float(result.get("table_leave_traveled_m", 0.0)) < TABLE_LEAVE_MIN_TRAVELED_M:
                raise RuntimeError(
                    f"table leave traveled {result.get('table_leave_traveled_m')} m, "
                    f"need at least {TABLE_LEAVE_MIN_TRAVELED_M:.2f} m"
                )
            _face_yaw_holding(
                node, place_yaw, args.yaw_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                key_prefix="leave_face_west",
            )

        phase = "staging_nav"
        result["phase"] = phase
        if not skip_to_approach:
            _face_yaw_holding(
                node, place_yaw, args.yaw_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                key_prefix="staging_face_west",
            )
            staging_final = _navigate_holding(
                node, stage_pose, args.nav_timeout, MAX_STAGING_NAV_M, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
            )
            result["staging_final_base"] = staging_final.tolist()
            result["staging_completed"] = True
            _sleep_holding(node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open)

            phase = "shelf_raise"
            result["phase"] = phase
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            clearance_slide = slide_for_held_z(
                current_slide, context["held_center_base"][2], place_world[2] + args.place_clearance,
            )
            result["place_clearance_slide"] = clearance_slide
            _traverse_spine_holding(
                node, current_slide, clearance_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
                context["hold_left"], context["hold_right"], args.gripper_open, result,
            )
            context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, clearance_slide)
            result["held_center_base_at_clearance"] = context["held_center_base"].tolist()
            result["shelf_raise_completed"] = True

        phase = "shelf_approach"
        result["phase"] = phase
        approach_start = _odom_pose(node)
        place_stand_now = place_stand_from_goal(place_world, place_yaw, context["held_center_base"])
        place_pose = np.array([place_stand_now[0], place_stand_now[1], place_yaw], dtype=float)
        result["place_stand_xy"] = place_stand_now.tolist()
        try:
            approach_final = _drive_line(
                node, approach_start, place_pose, 1, args.shelf_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, position_tolerance=0.03, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                key_prefix="shelf_approach",
                held_center_base=context["held_center_base"],
                place_world=place_world,
                place_radius=args.approach_accept_radius,
            )
        except TimeoutError as exc:
            approach_final = _odom_pose(node)
            inside = box_inside_place_radius(
                approach_final, context["held_center_base"], place_world, args.approach_accept_radius,
            )
            result["shelf_approach_timeout_estimated_place_world"] = inside["held_world"]
            result["shelf_approach_timeout_xy_error_m"] = inside["xy_error_m"]
            depth = shelf_inward_ok(inside["held_world"], place_world)
            result["shelf_approach_timeout_outward_m"] = depth["outward_m"]
            if not inside["within_radius"] or not depth["deep_enough"]:
                raise
            result["shelf_approach_accepted_inside_radius"] = True
            result["shelf_approach_timeout_error"] = str(exc)
        result["place_final_base"] = approach_final.tolist()
        result["shelf_approach_completed"] = True
        _sleep_holding(node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open)
        ready = box_inside_place_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.approach_accept_radius,
        )
        result["pre_lower_estimated_place_world"] = ready["held_world"]
        result["pre_lower_xy_error_m"] = ready["xy_error_m"]
        pre_lower_depth = shelf_inward_ok(ready["held_world"], place_world)
        result["pre_lower_outward_m"] = pre_lower_depth["outward_m"]
        if not ready["within_radius"] or not pre_lower_depth["deep_enough"]:
            raise RuntimeError(
                f"box is still on the shelf lip (xy error {ready['xy_error_m']:.3f} m, "
                f"outward {pre_lower_depth['outward_m']:.3f} m); "
                f"need <= {args.approach_accept_radius:.2f} m and inward of {MAX_PLACE_OUTWARD_M:.2f} m"
            )

        phase = "place_lower"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        place_slide = slide_for_held_z(current_slide, context["held_center_base"][2], place_world[2])
        result["place_slide"] = place_slide
        _traverse_spine_holding(
            node, current_slide, place_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
            context["hold_left"], context["hold_right"], args.gripper_open, result,
        )
        context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, place_slide)
        estimated = held_center_world(_odom_pose(node), context["held_center_base"])
        error = placement_error(estimated, place_world, args.place_radius)
        result["estimated_place_world"] = estimated.tolist()
        result["place_xy_error_m"] = error["xy_error_m"]
        result["place_z_error_m"] = error["z_error_m"]
        result["place_within_radius"] = error["within_radius"]
        result["place_lower_completed"] = True
        _sleep_holding(node, 1.0, context["hold_left"], context["hold_right"], args.gripper_open)
        release_ready = box_inside_place_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.approach_accept_radius,
        )
        result["pre_release_xy_error_m"] = release_ready["xy_error_m"]
        release_depth = shelf_inward_ok(release_ready["held_world"], place_world)
        result["pre_release_outward_m"] = release_depth["outward_m"]
        if not release_ready["within_radius"] or not release_depth["deep_enough"]:
            raise RuntimeError(
                f"refusing to release on the shelf lip; xy error {release_ready['xy_error_m']:.3f} m, "
                f"outward {release_depth['outward_m']:.3f} m"
            )
        if not error["within_radius"]:
            raise RuntimeError(
                f"estimated place xy error {error['xy_error_m']:.4f} m exceeds radius {args.place_radius:.2f} m"
            )

        phase = "release"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        backend = MMK2KdlBackend()
        left_fk = backend.forward("l", current_slide, context["hold_left"])
        right_fk = backend.forward("r", current_slide, context["hold_right"])
        left_release_xyz, right_release_xyz = release_cartesian(
            left_fk[:3, 3], right_fk[:3, 3], args.release_spread,
        )
        release_plan = solve_bimanual_pose(
            current_slide, context["hold_left"], context["hold_right"],
            left_release_xyz, right_release_xyz, backend=backend,
        )
        result["release_plan"] = release_plan
        release_left = np.asarray(release_plan["left_joint_target"], dtype=float)
        release_right = np.asarray(release_plan["right_joint_target"], dtype=float)
        _traverse_pair(
            node, context["hold_left"], context["hold_right"], release_left, release_right,
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            args.gripper_open, args.gripper_open, True, result,
        )
        context["release_left"] = release_left
        context["release_right"] = release_right
        _sleep_holding(node, 0.8, release_left, release_right, args.gripper_open, require_hold=False)
        released = True
        result["released"] = True

        phase = "shelf_retreat"
        result["phase"] = phase
        retreat_start = _odom_pose(node)
        retreat_target = pose_offset(retreat_start, args.shelf_retreat, reverse=True)
        result["shelf_retreat_target"] = retreat_target.tolist()
        retreat_final = _drive_line(
            node, retreat_start, retreat_target, -1, args.retreat_timeout, result,
            release_left, release_right, args.gripper_open,
            require_hold=False, min_traveled_m=max(0.20, args.shelf_retreat - 0.08),
            position_tolerance=0.06,
            max_linear_speed=MAX_SHELF_LINEAR_SPEED, key_prefix="shelf_retreat",
        )
        result["shelf_retreat_final_base"] = retreat_final.tolist()
        result["shelf_retreat_completed"] = True

        phase = "retract"
        result["phase"] = phase
        _retract_arms(node, context, args, result)
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["phase"] = phase
        if command_issued and node is not None:
            _recover(node, context, args, result, start_base, place_yaw, phase, released, place_world)
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
            try:
                node.controller.stop_base()
            except Exception:
                pass
            node.close(stop_robot=False)


if __name__ == "__main__":
    raise SystemExit(main())
