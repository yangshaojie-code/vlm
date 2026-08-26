"""Task 3 bounded yellow-box cube-top pick and shelf L1 place check.

Grabs the yellow box from the top of the white cube with the verified
bimanual hug geometry at the raised cube-top height, lifts 0.10 m clear of
the cube, backs off the table, and places it on shelf L1 left of the white
packaging-box obstacle.  It never enables the formal executor or returns to
the end zone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0, solve_bimanual_hug_pose, solve_bimanual_pose
from color_box_detector import detect_colored_boxes
from depth_utils import robust_depth_from_bbox
from geometry_utils import transform_point
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
    RETRACTION_JOINT_TOLERANCE_RAD,
    TASK1_APPROACH_HALF_M,
    TASK1_GRASP_FWD_OFFSET_M,
    TASK1_GRASP_Z_OFFSET_M,
    TASK1_HOLD_HALF_M,
    _approach_until_reached_or_contact,
    blocked_table_hug_lock,
    carry_hold_ok,
    contact_approach_geometry,
    contact_clearance_schedule,
    hug_moved_from_pregrasp,
    lift_slide_target,
)
from task1_precontact_check import (
    BOX_HALF_DEPTH,
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _navigate,
    _traverse_pair,
    _world_to_base,
    station_target,
    wrap_to_pi,
)
from task1_shelf_place_check import (
    DEFAULT_LINE_TIMEOUT_SEC,
    DEFAULT_NAV_TIMEOUT_SEC,
    DEFAULT_SHELF_TIMEOUT_SEC,
    DEFAULT_YAW_TIMEOUT_SEC,
    MAX_HOLD_LINEAR_SPEED,
    MAX_PLACE_OUTWARD_M,
    MAX_SHELF_LINEAR_SPEED,
    MAX_STAGING_NAV_M,
    SHELF_RETREAT_M,
    SHELF_ZONE_X_MAX,
    STAGING_BACK_M,
    TABLE_LEAVE_DISTANCE_M,
    TABLE_LEAVE_MIN_TRAVELED_M,
    _face_yaw_holding,
    _navigate_holding,
    _retract_arms,
    _sleep_holding,
    apply_slide_keep_hold,
    held_center_world,
    pose_offset,
    release_cartesian,
    slide_for_held_z,
    staging_pose,
)
from task1_transport_check import (
    CARRY_CLEARANCE_MAX_M,
    DEFAULT_SQUEEZE_SECONDS,
    _command_hug,
    _confirm_inward_squeeze,
    _odom_pose,
    _traverse_spine_holding,
)
from task2_shelf_to_table_check import (
    _bind_hold_keeper,
    _traverse_spine_keeping_pose,
    already_carrying_box,
    held_center_from_palms,
    inward_hold_from_blocked,
    local_carry_hold,
)


TASK_COLOR = "yellow"
YELLOW_FIXED_WORLD = np.array([-0.54, 2.30, 1.004], dtype=float)
CUBE_TOP_Z = 0.909
BOX_HALF_HEIGHT_M = 0.095
CUBE_TOP_CENTER_Z = CUBE_TOP_Z + BOX_HALF_HEIGHT_M
INSTRUCTION_PLACE_WORLD = np.array([-2.68, 0.54, 0.498], dtype=float)
PLACE_RADIUS_M = 0.24
PLACE_ACCEPT_RADIUS_M = 0.08
PLACE_YAW = math.pi
GRASP_YAW = math.pi / 2.0
TASK3_LIFT_HEIGHT_M = 0.10
TASK3_PLACE_CLEARANCE_M = 0.055
TASK3_RELEASE_SPREAD_M = 0.04
TASK3_STANDOFF_M = 0.54
SHELF_L1_BOARD_Z = 0.403
# The head camera sits ~0.35 m above the yellow box at the observe distance,
# so a small extra pitch (the body already tilts -0.33) keeps it centered.
OBSERVE_HEAD = (0.0, -0.10)
TABLE_SOUTH_EDGE_Y = 1.915
STATION_Y_MAX = TABLE_SOUTH_EDGE_Y - 0.03
HUG_WINDOW_X = (0.35, 0.75)
HUG_WINDOW_ABS_Y = 0.15
MAX_STATION_NAV_M = 2.60
APPROACH_STALL_SEC = 4.0


def validate_yellow_world(box_world) -> np.ndarray:
    """Reject detections that are not the fixed-layout cube-top yellow cell."""
    box = np.asarray(box_world, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)):
        raise ValueError("yellow box_world must be a finite [x, y, z] vector")
    if not (-0.80 <= box[0] <= -0.30 and 2.05 <= box[1] <= 2.55 and 0.94 <= box[2] <= 1.07):
        raise ValueError(f"yellow box is outside the Task 3 cube-top window: {box.tolist()}")
    return box


def snap_cube_top_center(box_world) -> np.ndarray:
    """Snap a cube-top detection onto the known fixed-layout yellow cell.

    The head camera looks north, so depth noise lands on world Y.  A 12 cm
    north bias makes station_target sit on the table-edge guard and abort
    before the hug.  The white cube pose is fixed, so XY and Z are snapped
    after the detection has been validated in the cube-top window.
    """
    box = validate_yellow_world(box_world).copy()
    box[:2] = YELLOW_FIXED_WORLD[:2]
    box[2] = CUBE_TOP_CENTER_Z
    return box


def center_from_cube_surface(
    surface_world,
    yaw: float = GRASP_YAW,
    half_depth: float = BOX_HALF_DEPTH,
    center_z: float = CUBE_TOP_CENTER_Z,
) -> np.ndarray:
    """Map an RGB-D hit on the cube-top box front face to its center."""
    surface = np.asarray(surface_world, dtype=float)
    yaw = wrap_to_pi(yaw)
    depth = float(half_depth)
    height = float(center_z)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError("surface must be a finite 3-vector")
    if not 0.05 <= depth <= 0.14:
        raise ValueError("cube-top half-depth must be within [0.05, 0.14] m")
    center = surface.copy()
    center[:2] += depth * np.array([math.cos(yaw), math.sin(yaw)])
    center[2] = height
    return center


def validate_place_world_l1(place_world) -> np.ndarray:
    """Reject place targets that are not the fixed-layout Task 3 L1 cell."""
    place = np.asarray(place_world, dtype=float)
    if place.shape != (3,) or not np.all(np.isfinite(place)):
        raise ValueError("place_world must be a finite [x, y, z] vector")
    if not (-2.95 <= place[0] <= -2.40 and 0.30 <= place[1] <= 0.72 and 0.42 <= place[2] <= 0.60):
        raise ValueError(f"place_world is outside the Task 3 shelf L1 window: {place.tolist()}")
    return place


def lift_clears_cube(box_z: float, lift_height: float, cube_top: float = CUBE_TOP_Z) -> bool:
    """True when a lifted cube-top box bottom clears the cube top with margin."""
    bottom_after_lift = float(box_z) - BOX_HALF_HEIGHT_M + float(lift_height)
    return bottom_after_lift >= float(cube_top) + 0.04


def station_for_yellow(
    box_world,
    standoff: float = TASK3_STANDOFF_M,
    yaw: float = GRASP_YAW,
) -> np.ndarray:
    """South-of-table station that puts the cube-top box into the hug window."""
    stand = station_target(box_world, standoff, yaw).copy()
    if float(stand[1]) > TABLE_SOUTH_EDGE_Y - 0.01:
        raise ValueError(f"station {stand.tolist()} is not south of the table edge")
    if float(stand[1]) > STATION_Y_MAX:
        stand[1] = STATION_Y_MAX
    return stand


def place_stand_from_goal(place_world, place_yaw: float, held_center_base) -> np.ndarray:
    """Base xy that puts the held box center onto place_world xy at place_yaw."""
    place = validate_place_world_l1(place_world)
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


def task3_placement_error(held_world, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Local xy/z error of the locked box versus the instruction place point."""
    held = np.asarray(held_world, dtype=float)
    place = validate_place_world_l1(place_world)
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
    error = task3_placement_error(world, place_world, place_radius)
    error["held_world"] = world.tolist()
    return error


def shelf_inward_ok(held_world, place_world, max_outward_m: float = MAX_PLACE_OUTWARD_M) -> dict:
    """True when the box is not still hanging off the west-facing shelf lip."""
    held = np.asarray(held_world, dtype=float)
    place = validate_place_world_l1(place_world)
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


def locate_yellow(node, yaw: float = GRASP_YAW):
    """RGB-D lock of the cube-top yellow front face, mapped to the box center."""
    snapshot = node.wait_for_snapshot(timeout_sec=4.0)
    detections = detect_colored_boxes(
        snapshot.rgb,
        TASK_COLOR,
        min_area=max(60, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 5000),
    )
    if not detections:
        raise RuntimeError("no yellow box detected in the current RGB frame")
    detection = max(detections, key=lambda item: item.area * item.confidence)
    depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
    camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
    frame = snapshot.camera_frame or "head_camera"
    camera_to_world = node.transforms.lookup("odom", frame)
    surface_world = transform_point(camera_to_world, camera_point)
    center_raw = validate_yellow_world(center_from_cube_surface(surface_world, yaw))
    center_world = snap_cube_top_center(center_raw)
    center_base = _world_to_base(node, center_world)
    return {
        "bbox": list(detection.bbox),
        "pixel": list(detection.center),
        "depth_m": float(depth),
        "surface_world": surface_world.tolist(),
        "center_world_raw": center_raw.tolist(),
        "center_world": center_world.tolist(),
        "center_base": center_base.tolist(),
        "source": "vision",
    }


def _look_at_table(node, result):
    node.controller.command_head(list(OBSERVE_HEAD))
    result["published_control_topics"] = list(dict.fromkeys(
        result.get("published_control_topics", []) + ["/head_forward_position_controller/commands"]
    ))
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        node.spin_once(0.05)


def _record_topic(result, topic: str) -> None:
    result["published_control_topics"] = list(dict.fromkeys(result.get("published_control_topics", []) + [topic]))


def _drive_line_task3(
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
    position_tolerance: float = 0.02,
    min_traveled_m: float = 0.0,
    max_linear_speed: float = MAX_SHELF_LINEAR_SPEED,
    key_prefix: str = "line",
    held_center_base=None,
    place_world=None,
    place_radius: float | None = None,
    hold_keeper=None,
):
    """Heading-aligned segment with the Task 3 L1 place accept checks."""
    from task1_shelf_place_check import held_line_command

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
        if hold_keeper is not None:
            left_joints, right_joints, contact = hold_keeper(
                left_current, right_current, left_joints, right_joints,
            )
            _command_hug(node, left_joints, right_joints, gripper_open)
            result[f"{key_prefix}_contact_feedback"] = contact
            if require_hold and not contact["holding"]:
                node.controller.stop_base()
                raise RuntimeError(f"held-box contact changed during {key_prefix}: {contact}")
        elif require_hold:
            _command_hug(node, left_joints, right_joints, gripper_open)
            contact = carry_hold_ok(left_current, right_current, left_joints, right_joints)
            result[f"{key_prefix}_contact_feedback"] = contact
            if not contact["holding"]:
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


def _establish_cube_top_hold(node, args, box_base, result):
    """Reproduce the verified Task 1 hug at the raised cube-top height."""
    initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
    initial_left_gripper = command_gripper_value(raw_left_gripper)
    initial_right_gripper = command_gripper_value(raw_right_gripper)
    initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    if initial_slide > 0.10:
        raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
    contact_slide = float(PRE_GRASP_Z0 - (box_base[2] + args.grasp_z_offset))
    if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
        raise RuntimeError(f"contact slide {contact_slide:.4f} is outside {SLIDE_LIMITS}")
    lift_slide = lift_slide_target(contact_slide, args.lift_height)
    if not lift_clears_cube(float(box_base[2]), args.lift_height):
        raise RuntimeError("lift height does not clear the white cube top")
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
        left_reference = np.asarray(plan["left_joint_target"])
        right_reference = np.asarray(plan["right_joint_target"])
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
        "hold_left": None,
        "hold_right": None,
        "held_center_base": None,
    }
    high_left, high_right = _traverse_pair(
        node, initial_left, initial_right, high_plan["left_joint_target"], high_plan["right_joint_target"],
        args.joint_max_step, args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD,
        initial_left_gripper, initial_right_gripper, True, result,
    )
    context["high_left"], context["high_right"] = high_left, high_right
    open_left, open_right = _traverse_grippers(
        node, high_left, high_right, initial_left_gripper, initial_right_gripper,
        args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result,
    )
    result["reached_open_left_gripper"] = open_left
    result["reached_open_right_gripper"] = open_right
    # Arms stay at the wide open pose while the slide drops to cube-top height.
    _traverse_spine_keeping_pose(
        node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        high_left, high_right, args.gripper_open, result,
    )
    reached_left, reached_right = high_left, high_right
    reached = []
    contact_plan = None
    contact_feedback = None
    hold_left = hold_right = None
    for plan_index, plan in enumerate(plans):
        result["phase"] = f"approach_clearance_{plan['clearance_m']:.3f}"
        try:
            reached_left, reached_right, waypoint_result = _approach_until_reached_or_contact(
                node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.gripper_open, args.gripper_open,
                args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD, result,
                allow_early_contact=plan["clearance_m"] <= CARRY_CLEARANCE_MAX_M,
            )
        except TimeoutError as exc:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            locked = blocked_table_hug_lock(
                left_now, right_now, high_left, high_right,
                plan["left_joint_target"], plan["right_joint_target"],
            )
            if locked is None:
                raise
            reached_left, reached_right = locked["left"], locked["right"]
            waypoint_result = locked["feedback"]
            result["blocked_hug_lock"] = {
                "clearance_m": plan["clearance_m"],
                "left": reached_left.tolist(),
                "right": reached_right.tolist(),
                "timeout_error": str(exc),
                "left_max_joint_residual_rad": waypoint_result["left_max_joint_residual_rad"],
                "right_max_joint_residual_rad": waypoint_result["right_max_joint_residual_rad"],
            }
        reached.append({
            "clearance_m": plan["clearance_m"],
            "left": np.asarray(reached_left, dtype=float).tolist(),
            "right": np.asarray(reached_right, dtype=float).tolist(),
            "contact_detected": bool(waypoint_result.get("contact_detected")),
            "blocked_hug": bool(waypoint_result.get("blocked_hug")),
        })
        if not waypoint_result.get("contact_detected"):
            continue
        if not hug_moved_from_pregrasp(reached_left, reached_right, high_left, high_right):
            result.setdefault("rejected_false_contacts", []).append({
                "clearance_m": plan["clearance_m"], "reason": "still_at_pregrasp",
            })
            continue
        if waypoint_result.get("blocked_hug"):
            tight = plans[-1]
            hold_left, hold_right = inward_hold_from_blocked(
                reached_left, reached_right, tight["left_joint_target"], tight["right_joint_target"],
            )
            try:
                squeeze = _confirm_inward_squeeze(
                    node, hold_left, hold_right, args.gripper_open, args.squeeze_seconds, result,
                )
                result["squeeze_confirmed"] = True
                result["squeeze_feedback"] = squeeze
            except RuntimeError as exc:
                result["blocked_hug_squeeze_error"] = str(exc)
            contact_plan = plan
            contact_feedback = waypoint_result
            break
        if plan["clearance_m"] > CARRY_CLEARANCE_MAX_M:
            result.setdefault("rejected_false_contacts", []).append({
                "clearance_m": plan["clearance_m"], "reason": "too_open_to_carry",
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
        result["squeeze_confirmed"] = True
        result["squeeze_feedback"] = squeeze
        contact_plan = plan
        contact_feedback = waypoint_result
        break
    if contact_plan is None:
        raise TimeoutError("dual-arm contact was not established at any validated clearance waypoint")
    result["contact_detected"] = True
    result["contact_feedback"] = contact_feedback
    result["contact_clearance_detected_m"] = contact_plan["clearance_m"]
    result["reached_contact_plans"] = reached
    result["hold_joint_targets"] = {"left": np.asarray(hold_left).tolist(), "right": np.asarray(hold_right).tolist()}
    hold_left = np.asarray(hold_left, dtype=float)
    hold_right = np.asarray(hold_right, dtype=float)
    # The cube top is 0.10 m below the lifted box bottom.  Leaving the table
    # at contact height drags the yellow box off the white cube.
    _traverse_spine_holding(
        node, contact_slide, lift_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        hold_left, hold_right, args.gripper_open, result,
    )
    result["lift_completed"] = True
    result["lift_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0])
    context.update({
        "hold_left": hold_left,
        "hold_right": hold_right,
        "held_center_base": apply_slide_keep_hold(box_base, contact_slide, lift_slide),
    })
    return context


def _recover(node, context, args, result, start_base, phase: str, released: bool, place_world):
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
    if holding and near_shelf:
        result["recovery_kept_hold"] = True
        result["recovery_skipped_pull_out"] = True
        try:
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
        except Exception as exc:
            result["recovery_keep_hold_error"] = str(exc)
    elif holding and phase in {"hug_lift", "table_leave"} and start_base is not None and current is not None:
        try:
            _drive_line_task3(
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
    elif released and near_shelf and current is not None:
        try:
            _drive_line_task3(
                node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                args.retreat_timeout, result,
                context.get("release_left", context.get("hold_left")),
                context.get("release_right", context.get("hold_right")), args.gripper_open,
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
    parser = argparse.ArgumentParser(description="Task 3 cube-top hug-to-shelf L1 placement check")
    parser.add_argument("--box-world", nargs=3, type=float, help="optional cube-top yellow center override")
    parser.add_argument("--no-allow-fixed-yellow", action="store_false", dest="allow_fixed_yellow")
    parser.set_defaults(allow_fixed_yellow=True)
    parser.add_argument("--place-world", nargs=3, type=float, default=INSTRUCTION_PLACE_WORLD.tolist())
    parser.add_argument("--place-radius", type=float, default=PLACE_RADIUS_M)
    parser.add_argument("--place-accept-radius", type=float, default=PLACE_ACCEPT_RADIUS_M)
    parser.add_argument("--place-yaw", type=float, default=PLACE_YAW)
    parser.add_argument("--grasp-yaw", type=float, default=GRASP_YAW)
    parser.add_argument("--standoff", type=float, default=TASK3_STANDOFF_M)
    parser.add_argument("--initial-clearance", type=float, default=0.02)
    parser.add_argument("--contact-step", type=float, default=0.01)
    parser.add_argument("--grasp-fwd-offset", type=float, default=TASK1_GRASP_FWD_OFFSET_M)
    parser.add_argument("--grasp-z-offset", type=float, default=TASK1_GRASP_Z_OFFSET_M)
    parser.add_argument("--approach-half", type=float, default=TASK1_APPROACH_HALF_M)
    parser.add_argument("--hold-half", type=float, default=TASK1_HOLD_HALF_M)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-max-step", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=TASK3_LIFT_HEIGHT_M)
    parser.add_argument("--table-leave-distance", type=float, default=TABLE_LEAVE_DISTANCE_M)
    parser.add_argument("--place-clearance", type=float, default=TASK3_PLACE_CLEARANCE_M)
    parser.add_argument("--release-spread", type=float, default=TASK3_RELEASE_SPREAD_M)
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
    parser.add_argument("--output", default="/tmp/task3_cube_top_shelf_place_check.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.0 <= args.gripper_open <= 1.0:
        parser.error("gripper-open is invalid")
    if args.settle_timeout <= 0 or args.nav_timeout <= 0 or args.yaw_timeout <= 0 or args.squeeze_seconds <= 0:
        parser.error("timeout arguments are invalid")
    if not TASK1_HOLD_HALF_M <= args.hold_half <= args.approach_half <= TASK1_APPROACH_HALF_M:
        parser.error("hold-half must be within [0.115, approach-half], and approach-half <= 0.13 m")
    if not 0.06 <= args.place_accept_radius <= args.place_radius:
        parser.error("place-accept-radius must be within [0.06, place-radius]")
    if not lift_clears_cube(YELLOW_FIXED_WORLD[2], args.lift_height):
        parser.error("lift height would not clear the white cube top")
    place_world = validate_place_world_l1(args.place_world)
    place_yaw = wrap_to_pi(args.place_yaw)
    grasp_yaw = wrap_to_pi(args.grasp_yaw)
    print(
        f"task3_cube_top_shelf_place_check starting apply={bool(args.apply)} output={args.output}",
        flush=True,
    )

    result = {
        "mode": "task3_bimanual_hug_cube_top_shelf_place_check",
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
        "place_accept_radius_m": args.place_accept_radius,
        "place_yaw_rad": place_yaw,
        "grasp_yaw_rad": grasp_yaw,
        "standoff_m": args.standoff,
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
    try:
        node = Ros2MissionNode(node_name="task3_cube_top_shelf_place_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
        initial_left_gripper = command_gripper_value(raw_left_gripper)
        initial_right_gripper = command_gripper_value(raw_right_gripper)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        start_base = _odom_pose(node)
        context.update({
            "initial_left": initial_left,
            "initial_right": initial_right,
            "initial_left_gripper": initial_left_gripper,
            "initial_right_gripper": initial_right_gripper,
            "initial_slide": initial_slide,
        })
        result["initial_base"] = start_base.tolist()
        result["initial_slide"] = initial_slide
        carrying = already_carrying_box(initial_slide, initial_left, initial_right)
        in_shelf = float(start_base[0]) <= SHELF_ZONE_X_MAX
        resume_transport = bool(carrying and not in_shelf)
        resume_shelf = bool(carrying and in_shelf)
        result["resumed_transport"] = resume_transport
        result["resumed_shelf"] = resume_shelf
        if resume_transport:
            print("resuming held transport; skipping cube-top grasp", flush=True)
        if resume_shelf:
            print("resuming in-shelf hug; skipping grasp and table leave", flush=True)

        phase = "detect"
        result["phase"] = phase
        keeper = _bind_hold_keeper(context, result)
        located = None
        box_world = None
        if args.box_world is not None:
            box_world = snap_cube_top_center(validate_yellow_world(args.box_world))
            located = {"center_world": box_world.tolist(), "source": "cli"}
        elif carrying:
            box_world = YELLOW_FIXED_WORLD.copy()
            located = {"center_world": box_world.tolist(), "source": "resume_carry"}
        elif not args.apply:
            box_world = YELLOW_FIXED_WORLD.copy()
            located = {"center_world": box_world.tolist(), "source": "fixed_layout_dry_run"}
        else:
            _look_at_table(node, result)
            try:
                located = locate_yellow(node, grasp_yaw)
            except Exception as exc:
                result["vision_error"] = str(exc)
                if not args.allow_fixed_yellow:
                    raise
                box_world = YELLOW_FIXED_WORLD.copy()
                located = {"center_world": box_world.tolist(), "source": "fixed_layout_fallback"}
        if box_world is None:
            box_world = snap_cube_top_center(located["center_world"])
        result["detection"] = located
        result["box_world_snapped"] = box_world.tolist()
        print(f"phase=detect detection_source={located.get('source')}", flush=True)

        station = station_for_yellow(box_world, args.standoff, grasp_yaw)
        result["station_target"] = station.tolist()
        place_stand = place_stand_from_goal(place_world, place_yaw, np.array([0.56, 0.0, box_world[2] + args.lift_height]))
        stage_pose = staging_pose(place_stand, place_yaw, args.staging_back)
        result["place_stand_xy_nominal"] = place_stand.tolist()
        result["staging_pose_nominal"] = stage_pose.tolist()

        contact_slide = float(PRE_GRASP_Z0 - (box_world[2] + args.grasp_z_offset))
        if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError(f"contact slide {contact_slide:.4f} is outside {SLIDE_LIMITS}")
        lift_slide = lift_slide_target(contact_slide, args.lift_height)
        result["contact_slide"] = contact_slide
        result["lift_slide"] = lift_slide
        held_after_lift = apply_slide_keep_hold(box_world, contact_slide, lift_slide)
        clearance_slide = slide_for_held_z(lift_slide, held_after_lift[2], place_world[2] + args.place_clearance)
        place_slide = slide_for_held_z(clearance_slide, apply_slide_keep_hold(held_after_lift, lift_slide, clearance_slide)[2], place_world[2])
        result["place_clearance_slide"] = clearance_slide
        result["place_slide"] = place_slide
        if max(clearance_slide, place_slide) > SLIDE_LIMITS[1] - 0.02:
            result["low_shelf_slide_margin_warning"] = True

        if not args.apply:
            result["status"] = "dry_run"
            result["box_contact_commanded"] = False
            result["base_motion_commanded"] = False
            result["transport_or_place_commanded"] = False
            print(
                "task3 dry-run ok (this is not a live pass); "
                f"contact_slide={contact_slide:.3f} lift_slide={lift_slide:.3f} "
                f"place_slide={place_slide:.3f} station={station.tolist()}",
                flush=True,
            )
            return 0

        command_issued = True
        if not carrying:
            phase = "station"
            result["phase"] = phase
            _navigate(
                node, station, 0.04, 0.06, args.nav_timeout, MAX_STATION_NAV_M,
                0.12, 0.50, result,
            )
            station_navigation = {
                "final_base": result.get("final_base"),
                "remaining_position_error_m": result.get("remaining_position_error_m"),
                "remaining_yaw_error_rad": result.get("remaining_yaw_error_rad"),
                "navigation_phase": result.get("navigation_phase"),
            }
            result["station_navigation"] = station_navigation
            box_base = np.asarray(_world_to_base(node, box_world), dtype=float)
            result["box_base_at_hug"] = box_base.tolist()
            if not (HUG_WINDOW_X[0] <= box_base[0] <= HUG_WINDOW_X[1] and abs(box_base[1]) <= HUG_WINDOW_ABS_Y):
                raise RuntimeError(f"yellow box is outside the hug window after station: {box_base.tolist()}")

            phase = "hug_lift"
            result["phase"] = phase
            context.update(_establish_cube_top_hold(node, args, box_base, result))
            context["contact_slide"] = result["contact_slide"]
            start_base = _odom_pose(node)
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            result["lift_completed"] = True
            _sleep_holding(
                node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
                hold_keeper=keeper,
            )

            phase = "table_leave"
            result["phase"] = phase
            table_leave_target = pose_offset(start_base, args.table_leave_distance, reverse=True)
            result["table_leave_target"] = table_leave_target.tolist()
            leave_final = _drive_line_task3(
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
        else:
            result["lift_completed"] = True
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            try:
                context["held_center_base"] = held_center_from_palms(current_slide, initial_left, initial_right)
                result["held_center_source"] = "palm_fk"
            except Exception as exc:
                result["palm_center_error"] = str(exc)
                context["held_center_base"] = np.asarray(_world_to_base(node, box_world), dtype=float)
                result["held_center_source"] = "snapped_vision"
            context["hold_left"], context["hold_right"] = local_carry_hold(initial_left, initial_right)
            result["hold_joint_targets"] = {
                "left": np.asarray(context["hold_left"]).tolist(),
                "right": np.asarray(context["hold_right"]).tolist(),
            }
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)

        phase = "face_west"
        result["phase"] = phase
        _face_yaw_holding(
            node, place_yaw, args.yaw_timeout, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            key_prefix="face_west", hold_keeper=keeper,
        )

        phase = "staging_nav"
        result["phase"] = phase
        place_stand = place_stand_from_goal(place_world, place_yaw, context["held_center_base"])
        stage_pose = staging_pose(place_stand, place_yaw, args.staging_back)
        result["place_stand_xy"] = place_stand.tolist()
        result["staging_pose"] = stage_pose.tolist()
        staging_final = _navigate_holding(
            node, stage_pose, args.nav_timeout, MAX_STAGING_NAV_M, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        result["staging_final_base"] = staging_final.tolist()
        result["staging_completed"] = True
        _sleep_holding(
            node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )

        phase = "shelf_lower"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        clearance_slide = slide_for_held_z(
            current_slide, context["held_center_base"][2], place_world[2] + args.place_clearance,
        )
        result["place_clearance_slide"] = clearance_slide
        _traverse_spine_holding(
            node, current_slide, clearance_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
            context["hold_left"], context["hold_right"], args.gripper_open, result,
            hold_keeper=keeper,
        )
        context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, clearance_slide)
        try:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            context["held_center_base"] = held_center_from_palms(
                float(node.sensors.joint_vector(["slide_joint"])[0]), left_now, right_now,
            )
            result["held_center_source_at_clearance"] = "palm_fk"
        except Exception as exc:
            result["clearance_palm_center_error"] = str(exc)
        result["held_center_base_at_clearance"] = context["held_center_base"].tolist()
        result["shelf_lower_completed"] = True

        phase = "shelf_approach"
        result["phase"] = phase
        approach_start = _odom_pose(node)
        place_stand_now = place_stand_from_goal(place_world, place_yaw, context["held_center_base"])
        place_pose = np.array([place_stand_now[0], place_stand_now[1], place_yaw], dtype=float)
        result["place_stand_xy"] = place_stand_now.tolist()
        try:
            approach_final = _drive_line_task3(
                node, approach_start, place_pose, 1, args.shelf_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, position_tolerance=0.03, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                key_prefix="shelf_approach",
                held_center_base=context["held_center_base"],
                place_world=place_world,
                place_radius=args.place_accept_radius,
                hold_keeper=keeper,
            )
        except TimeoutError as exc:
            approach_final = _odom_pose(node)
            inside = box_inside_place_radius(
                approach_final, context["held_center_base"], place_world, args.place_accept_radius,
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
        _sleep_holding(
            node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        ready = box_inside_place_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
        )
        result["pre_lower_estimated_place_world"] = ready["held_world"]
        result["pre_lower_xy_error_m"] = ready["xy_error_m"]
        pre_lower_depth = shelf_inward_ok(ready["held_world"], place_world)
        result["pre_lower_outward_m"] = pre_lower_depth["outward_m"]
        if not ready["within_radius"] or not pre_lower_depth["deep_enough"]:
            raise RuntimeError(
                f"box is still on the shelf lip (xy error {ready['xy_error_m']:.3f} m, "
                f"outward {pre_lower_depth['outward_m']:.3f} m); "
                f"need <= {args.place_accept_radius:.2f} m and inward of {MAX_PLACE_OUTWARD_M:.2f} m"
            )

        phase = "place_lower"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        place_slide = slide_for_held_z(current_slide, context["held_center_base"][2], place_world[2])
        result["place_slide"] = place_slide
        _traverse_spine_holding(
            node, current_slide, place_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
            context["hold_left"], context["hold_right"], args.gripper_open, result,
            hold_keeper=keeper,
        )
        context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, place_slide)
        estimated = held_center_world(_odom_pose(node), context["held_center_base"])
        error = task3_placement_error(estimated, place_world, args.place_radius)
        result["estimated_place_world"] = estimated.tolist()
        result["place_xy_error_m"] = error["xy_error_m"]
        result["place_z_error_m"] = error["z_error_m"]
        result["place_within_radius"] = error["within_radius"]
        result["place_lower_completed"] = True
        _sleep_holding(
            node, 1.0, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        release_ready = box_inside_place_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
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
        retreat_final = _drive_line_task3(
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
        result["failure_phase"] = phase
        print(f"task3_cube_top_shelf_place_check failed in {phase}: {exc}", flush=True)
        if command_issued and node is not None:
            try:
                _recover(node, context, args, result, start_base, phase, released, place_world)
            except Exception as recover_exc:
                result["recovery_fatal_error"] = str(recover_exc)
        return 2
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            print(f"could not write report {args.output}: {exc}", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        if node is not None:
            try:
                node.controller.stop_base()
            except Exception:
                pass
            try:
                node.close(stop_robot=False)
            except Exception:
                try:
                    node.destroy_node()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
