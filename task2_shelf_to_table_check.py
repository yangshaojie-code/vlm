"""Task 2 bounded brown-box shelf pick and table place check.

Starts from the current shelf-front pose after Task 1. It looks up the L2
brown box, hugs it at that layer (never raising arms through the pink L3
cell), reverses at least 0.20 m, then places onto the original table slot.
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
    CONTACT_MAX_JOINT_RESIDUAL_RAD,
    CONTACT_MIN_JOINT_RESIDUAL_RAD,
    RETRACTION_JOINT_TOLERANCE_RAD,
    _approach_until_reached_or_contact,
    carry_hold_ok,
    contact_clearance_schedule,
    contact_residuals,
    hug_moved_from_pregrasp,
    lift_slide_target,
)
from task1_precontact_check import (
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _navigate,
    _traverse_pair,
    _world_to_base,
    wrap_to_pi,
)
from task1_shelf_place_check import (
    DEFAULT_LINE_TIMEOUT_SEC,
    DEFAULT_NAV_TIMEOUT_SEC,
    DEFAULT_SHELF_TIMEOUT_SEC,
    DEFAULT_YAW_TIMEOUT_SEC,
    MAX_SHELF_LINEAR_SPEED,
    MAX_STAGING_NAV_M,
    PLACE_RELEASE_SPREAD_M,
    SHELF_RETREAT_M,
    SHELF_ZONE_X_MAX,
    STAGING_BACK_M,
    TABLE_LEAVE_MIN_TRAVELED_M,
    _drive_line,
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


TASK_COLOR = "brown"
BROWN_FIXED_WORLD = np.array([-2.63, 0.778, 0.837], dtype=float)
INSTRUCTION_PLACE_WORLD = np.array([-1.00, 2.20, 0.834], dtype=float)
PLACE_RADIUS_M = 0.28
PLACE_ACCEPT_RADIUS_M = 0.18
PICK_YAW = math.pi
PLACE_YAW = math.pi / 2.0
BOX_HALF_X_M = 0.12
BOX_HALF_Y_M = 0.08
BOX_HALF_Z_M = 0.095
SHELF_L2_BOARD_Z = 0.732
L2_BOX_CENTER_Z = 0.837
SHELF_L3_BOARD_Z = 1.061
SHELF_HOLD_HALF_M = 0.08
SHELF_APPROACH_HALF_M = 0.10
SHELF_GRASP_FWD_OFFSET_M = 0.065
SHELF_GRASP_Z_OFFSET_M = 0.045
SHELF_LIFT_HEIGHT_M = 0.08
# A shelf hug can stall 0.15-0.20 rad short of the planned 0.10/0.08 gap
# because the palms hit the box or cabinet before the IK target.  Task 1's
# 0.08 rad contact band would treat that as a settle failure and open the arms.
SHELF_BLOCKED_HUG_MIN_RAD = 0.04
SHELF_BLOCKED_HUG_MAX_RAD = 0.28
# Command this much tighter than the stalled joints so carry_hold_ok stays
# in-band.  A 0.22 blend toward a far IK target can exceed 0.08 rad.
CARRY_HOLD_SQUEEZE_RAD = 0.04
PICK_STANDOFF_M = 0.56
PICK_STAND_X_MIN = -1.96
TABLE_PLACE_CLEARANCE_M = 0.12
TABLE_RETREAT_M = 0.28
TABLE_RELEASE_SPREAD_M = 0.06
TABLE_RELEASE_RAISE_M = 0.05
# Table slab y∈[1.915, 2.715]. Drive closer than raw palm FK and refuse to
# lower until the estimated box center is clearly on the top.
TABLE_SOUTH_EDGE_Y = 1.915
TABLE_PLACE_Y_MIN = 2.02
TABLE_PLACE_HELD_X_MAX = 0.62
OBSERVE_HEAD = (0.0, -0.35)


def validate_brown_world(box_world) -> np.ndarray:
    """Reject detections that are not the fixed-layout L2 brown cell."""
    box = np.asarray(box_world, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)):
        raise ValueError("brown box_world must be a finite [x, y, z] vector")
    if not (-2.80 <= box[0] <= -2.40 and 0.50 <= box[1] <= 1.05 and 0.70 <= box[2] <= 0.98):
        raise ValueError(f"brown box is outside the Task 2 L2 window: {box.tolist()}")
    return box


def validate_table_place_world(place_world) -> np.ndarray:
    """Reject place targets that are not the Task 1 original table slot."""
    place = np.asarray(place_world, dtype=float)
    if place.shape != (3,) or not np.all(np.isfinite(place)):
        raise ValueError("place_world must be a finite [x, y, z] vector")
    if not (-1.30 <= place[0] <= -0.70 and 1.90 <= place[1] <= 2.50 and 0.75 <= place[2] <= 0.95):
        raise ValueError(f"place_world is outside the Task 2 table window: {place.tolist()}")
    return place


def center_from_shelf_front(surface_world, pick_yaw: float = PICK_YAW, half_depth: float = BOX_HALF_X_M) -> np.ndarray:
    """Map an RGB-D hit on the west-facing shelf front to the box center."""
    surface = np.asarray(surface_world, dtype=float)
    yaw = wrap_to_pi(pick_yaw)
    depth = float(half_depth)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError("surface must be a finite 3-vector")
    if not 0.06 <= depth <= 0.14:
        raise ValueError("shelf half-depth must be within [0.06, 0.14] m")
    center = surface.copy()
    center[:2] += depth * np.array([math.cos(yaw), math.sin(yaw)])
    return center


def shelf_hug_targets(box_base, clearance: float, grasp_fwd_offset: float, grasp_z_offset: float, hold_half: float):
    """Lateral hug targets for the 0.16 m shelf-face width."""
    box_base = np.asarray(box_base, dtype=float)
    clearance = float(clearance)
    hold_half = float(hold_half)
    if box_base.shape != (3,) or not np.all(np.isfinite(box_base)):
        raise ValueError("box_base must contain three finite values")
    if not 0.0 <= clearance <= 0.05:
        raise ValueError("contact clearance must be within [0.0, 0.05] m")
    if not 0.0 <= float(grasp_fwd_offset) <= 0.10 or not 0.0 <= float(grasp_z_offset) <= 0.10:
        raise ValueError("grasp offsets are outside the validated bounds")
    if not 0.07 <= hold_half <= 0.10:
        raise ValueError("shelf hold half gap must be within [0.07, 0.10] m")
    if not 0.30 <= box_base[0] <= 0.80 or abs(box_base[1]) > 0.18:
        raise ValueError(f"box is outside the safe base-frame approach window: {box_base.tolist()}")
    center = box_base + np.array([float(grasp_fwd_offset), 0.0, float(grasp_z_offset)])
    half = hold_half + clearance
    return center + np.array([0.0, half, 0.0]), center + np.array([0.0, -half, 0.0])


def pick_stand_from_box(box_world, pick_yaw: float, standoff: float = PICK_STANDOFF_M, x_min: float = PICK_STAND_X_MIN) -> np.ndarray:
    """Base pose that puts the L2 box in the hug window without driving into the cabinet."""
    box = validate_brown_world(box_world)
    yaw = wrap_to_pi(pick_yaw)
    standoff = float(standoff)
    if not 0.45 <= standoff <= 0.70:
        raise ValueError("pick standoff must be within [0.45, 0.70] m")
    stand = np.array([
        box[0] - standoff * math.cos(yaw),
        box[1] - standoff * math.sin(yaw),
        yaw,
    ], dtype=float)
    if stand[0] < float(x_min):
        stand[0] = float(x_min)
    return stand


def place_stand_xy(place_world, place_yaw: float, held_center_base) -> np.ndarray:
    """Base xy that puts the held box center onto the table place point."""
    place = validate_table_place_world(place_world)
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


def table_placement_error(held_world, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Local xy/z error of the locked box versus the table place point."""
    held = np.asarray(held_world, dtype=float)
    place = validate_table_place_world(place_world)
    radius = float(place_radius)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_world must be a finite 3-vector")
    if not 0.10 <= radius <= 0.30:
        raise ValueError("place radius must be within [0.10, 0.30] m")
    xy_error = float(np.linalg.norm(held[:2] - place[:2]))
    z_error = float(abs(held[2] - place[2]))
    return {
        "xy_error_m": xy_error,
        "z_error_m": z_error,
        "within_radius": xy_error <= radius,
    }


def box_inside_table_radius(base_pose, held_center_base, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Judge the locked box xy against the table place cylinder."""
    world = held_center_world(base_pose, held_center_base)
    error = table_placement_error(world, place_world, place_radius)
    error["held_world"] = world.tolist()
    return error


def local_carry_hold(reached_left, reached_right, squeeze_rad: float = CARRY_HOLD_SQUEEZE_RAD):
    """Keep the current hug and ask j2 a little tighter so residual stays in-band."""
    left = np.asarray(reached_left, dtype=float).copy()
    right = np.asarray(reached_right, dtype=float).copy()
    squeeze_rad = float(squeeze_rad)
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError("carry-hold vectors must be six-joint arrays")
    if not CONTACT_MIN_JOINT_RESIDUAL_RAD <= squeeze_rad <= CONTACT_MAX_JOINT_RESIDUAL_RAD:
        raise ValueError("carry-hold squeeze must stay inside the contact residual band")
    left[1] -= squeeze_rad
    right[1] -= squeeze_rad
    return left, right


def maintain_carry_hold(left_current, right_current, left_hold, right_hold):
    """Keep asking a 0.04 rad tighter hug if the previous squeeze has settled."""
    left_current = np.asarray(left_current, dtype=float)
    right_current = np.asarray(right_current, dtype=float)
    left_hold = np.asarray(left_hold, dtype=float)
    right_hold = np.asarray(right_hold, dtype=float)
    contact = carry_hold_ok(left_current, right_current, left_hold, right_hold)
    if contact["holding"]:
        return left_hold, right_hold, contact
    left_err = float(contact["left_max_joint_residual_rad"])
    right_err = float(contact["right_max_joint_residual_rad"])
    hugging = float(left_current[1]) <= -0.70 and float(right_current[1]) <= -0.70
    settled = (
        hugging
        and left_err <= CONTACT_MAX_JOINT_RESIDUAL_RAD + 0.02
        and right_err <= CONTACT_MAX_JOINT_RESIDUAL_RAD + 0.02
    )
    if not settled:
        return left_hold, right_hold, contact
    new_left, new_right = local_carry_hold(left_current, right_current)
    refreshed = carry_hold_ok(left_current, right_current, new_left, new_right)
    refreshed["hold_refreshed"] = True
    return new_left, new_right, refreshed


def _bind_hold_keeper(context, result):
    def keeper(left_current, right_current, left_hold, right_hold):
        new_left, new_right, contact = maintain_carry_hold(
            left_current, right_current, left_hold, right_hold,
        )
        context["hold_left"] = np.asarray(new_left, dtype=float)
        context["hold_right"] = np.asarray(new_right, dtype=float)
        if contact.get("hold_refreshed"):
            result["hold_refresh_count"] = int(result.get("hold_refresh_count", 0)) + 1
        return context["hold_left"], context["hold_right"], contact
    return keeper


def inward_hold_from_blocked(reached_left, reached_right, tight_left, tight_right, blend: float = 0.22):
    """Ask for a slightly tighter pose than the stalled hug so carry_hold_ok stays in-band."""
    reached_left = np.asarray(reached_left, dtype=float)
    reached_right = np.asarray(reached_right, dtype=float)
    tight_left = np.asarray(tight_left, dtype=float)
    tight_right = np.asarray(tight_right, dtype=float)
    blend = float(blend)
    if reached_left.shape != (6,) or reached_right.shape != (6,) or tight_left.shape != (6,) or tight_right.shape != (6,):
        raise ValueError("blocked-hug hold vectors must be six-joint arrays")
    if not 0.10 <= blend <= 0.40:
        raise ValueError("blocked-hug blend must be within [0.10, 0.40]")
    delta_left = tight_left - reached_left
    delta_right = tight_right - reached_right
    max_abs = float(max(np.max(np.abs(delta_left)), np.max(np.abs(delta_right)), 1e-9))
    if max_abs < 0.02:
        return local_carry_hold(reached_left, reached_right)
    scale = min(blend, CARRY_HOLD_SQUEEZE_RAD / max_abs)
    return reached_left + scale * delta_left, reached_right + scale * delta_right


def held_center_from_palms(slide: float, left_joints, right_joints) -> np.ndarray:
    """Box center estimate from the current palm FK, used when resuming a hug."""
    backend = MMK2KdlBackend()
    left_fk = backend.forward("l", float(slide), np.asarray(left_joints, dtype=float))
    right_fk = backend.forward("r", float(slide), np.asarray(right_joints, dtype=float))
    center = 0.5 * (np.asarray(left_fk[:3, 3], dtype=float) + np.asarray(right_fk[:3, 3], dtype=float))
    center[0] -= SHELF_GRASP_FWD_OFFSET_M
    if not (0.35 <= center[0] <= 0.85 and abs(center[1]) <= 0.18 and 0.70 <= center[2] <= 1.20):
        raise ValueError(f"palm midpoint is outside the hold window: {center.tolist()}")
    return center


def table_place_held_center(held_center_base) -> np.ndarray:
    """Cap forward reach so the base drives onto the table instead of stopping short."""
    held = np.asarray(held_center_base, dtype=float).copy()
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_center_base must be a finite 3-vector")
    held[0] = min(float(held[0]), TABLE_PLACE_HELD_X_MAX)
    return held


def box_on_table_top(held_world, y_min: float = TABLE_PLACE_Y_MIN) -> bool:
    """True when the estimated box center is north of the table's south edge."""
    world = np.asarray(held_world, dtype=float)
    if world.shape != (3,) or not np.all(np.isfinite(world)):
        return False
    return float(world[1]) >= float(y_min) and -1.37 <= float(world[0]) <= 0.29


def blocked_hug_lock(left_current, right_current, left_pregrasp, right_pregrasp, left_plan, right_plan):
    """Lock the current joints when both arms are stalled against the shelf box."""
    if not hug_moved_from_pregrasp(left_current, right_current, left_pregrasp, right_pregrasp):
        return None
    residual = contact_residuals(left_current, right_current, left_plan, right_plan)
    left_error = residual["left_max_joint_residual_rad"]
    right_error = residual["right_max_joint_residual_rad"]
    if not (SHELF_BLOCKED_HUG_MIN_RAD <= left_error <= SHELF_BLOCKED_HUG_MAX_RAD):
        return None
    if not (SHELF_BLOCKED_HUG_MIN_RAD <= right_error <= SHELF_BLOCKED_HUG_MAX_RAD):
        return None
    residual["blocked_hug"] = True
    residual["contact_detected"] = True
    residual["holding"] = True
    return {
        "left": np.asarray(left_current, dtype=float),
        "right": np.asarray(right_current, dtype=float),
        "feedback": residual,
    }


def lifted_box_clears_upper_board(box_z: float, lift_height: float, board_z: float = SHELF_L3_BOARD_Z) -> bool:
    """True when a lifted L2 box top stays below the L3 board."""
    top = float(box_z) + BOX_HALF_Z_M + float(lift_height)
    return top < float(board_z) - 0.02


def snap_l2_box_center(box_world) -> np.ndarray:
    """Keep XY from vision, but use the L2 layer center z.

    Side-view RGB-D often hits the top-front edge (~0.93 m) instead of the
    box center (0.837 m).  Using the raw z makes an 0.08 m lift look like it
    would strike the occupied L3 board.
    """
    box = validate_brown_world(box_world).copy()
    box[2] = L2_BOX_CENTER_Z
    return box


def already_carrying_box(slide, left_joints, right_joints) -> bool:
    """True when both arms are still in a closed shelf-hug at carry height."""
    left = np.asarray(left_joints, dtype=float)
    right = np.asarray(right_joints, dtype=float)
    if left.shape != (6,) or right.shape != (6,):
        return False
    return float(slide) >= 0.20 and float(left[1]) <= -0.70 and float(right[1]) <= -0.70


def already_at_table_place(base_pose) -> bool:
    """True when the base is already at the north table face, hugging a placed box."""
    base = np.asarray(base_pose, dtype=float)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        return False
    return float(base[1]) >= 1.40 and -1.40 <= float(base[0]) <= -0.70


def already_holding_on_shelf(base_pose, slide, left_joints, right_joints) -> bool:
    """True when a previous P4 attempt is still hugging the L2 box in the cabinet."""
    base = np.asarray(base_pose, dtype=float)
    if base.shape != (3,):
        return False
    return float(base[0]) <= SHELF_ZONE_X_MAX and already_carrying_box(slide, left_joints, right_joints)


def already_on_pick_approach(current_pose, pick_stand, staging) -> bool:
    """True when the base is already between staging and the clamped pick stand."""
    current = np.asarray(current_pose, dtype=float)
    stand = np.asarray(pick_stand, dtype=float)
    stage = np.asarray(staging, dtype=float)
    if current[0] > max(stand[0], stage[0]) + 0.12:
        return False
    if abs(current[1] - stand[1]) > 0.12:
        return False
    return current[0] <= stage[0] + 0.05


def locate_brown(node, pick_yaw: float = PICK_YAW):
    """RGB-D lock of the L2 brown front face, mapped to the box center."""
    snapshot = node.wait_for_snapshot(timeout_sec=4.0)
    detections = detect_colored_boxes(
        snapshot.rgb,
        TASK_COLOR,
        min_area=max(60, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 5000),
    )
    if not detections:
        raise RuntimeError("no brown box detected in the current RGB frame")
    detection = max(detections, key=lambda item: item.area * item.confidence)
    depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
    camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
    frame = snapshot.camera_frame or "head_camera"
    camera_to_world = node.transforms.lookup("odom", frame)
    surface_world = transform_point(camera_to_world, camera_point)
    center_raw = validate_brown_world(center_from_shelf_front(surface_world, pick_yaw))
    center_world = snap_l2_box_center(center_raw)
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


def _look_at_shelf(node, result):
    node.controller.command_head(list(OBSERVE_HEAD))
    result["published_control_topics"] = list(dict.fromkeys(
        result.get("published_control_topics", []) + ["/head_forward_position_controller/commands"]
    ))
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        node.spin_once(0.05)


def _traverse_spine_keeping_pose(node, start, end, max_step, timeout, tolerance, left_joints, right_joints, gripper_open, result):
    """Move the slide while holding a known empty-hand pose. Do not require box contact."""
    start, end = float(start), float(end)
    count = max(1, int(math.ceil(abs(end - start) / float(max_step))))
    result["spine_waypoint_count"] = count
    result["published_control_topics"] = list(dict.fromkeys(
        result.get("published_control_topics", []) + [
            "/spine_forward_position_controller/commands",
            "/left_arm_forward_position_controller/commands",
            "/right_arm_forward_position_controller/commands",
        ]
    ))
    reached = start
    for index in range(1, count + 1):
        target = start + index / count * (end - start)
        node.controller.command_spine(target)
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            node.spin_once(0.05)
            _command_hug(node, left_joints, right_joints, gripper_open)
            reached = float(node.sensors.joint_vector(["slide_joint"])[0])
            if abs(reached - target) <= float(tolerance):
                break
        else:
            raise TimeoutError("slide feedback did not reach waypoint with empty hands")
    return reached


def _close_shelf_hug(node, args, box_base, open_left, open_right, contact_slide, result):
    """Inward hug from the layer-matched open pose, then a short L2 lift."""
    plans = []
    left_reference = np.asarray(open_left, dtype=float)
    right_reference = np.asarray(open_right, dtype=float)
    clearances = contact_clearance_schedule(args.initial_clearance, args.contact_step)
    for clearance in clearances:
        half = args.hold_half if clearance == 0.0 else args.approach_half
        left_target, right_target = shelf_hug_targets(
            box_base, 0.0, args.grasp_fwd_offset, args.grasp_z_offset, half,
        )
        plan = solve_bimanual_pose(contact_slide, left_reference, right_reference, left_target, right_target)
        plan["clearance_m"] = clearance
        plan["half_gap_m"] = half
        plans.append(plan)
        left_reference = np.asarray(plan["left_joint_target"])
        right_reference = np.asarray(plan["right_joint_target"])
    result["contact_plans"] = plans
    result["contact_clearance_schedule_m"] = clearances
    lift_slide = lift_slide_target(contact_slide, args.lift_height)
    result["lift_slide"] = lift_slide
    reached_left, reached_right = np.asarray(open_left, dtype=float), np.asarray(open_right, dtype=float)
    reached = []
    contact_plan = None
    contact_feedback = None
    hold_left = hold_right = None
    for plan in plans:
        try:
            reached_left, reached_right, waypoint_result = _approach_until_reached_or_contact(
                node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.gripper_open, args.gripper_open,
                args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD, result,
                allow_early_contact=True,
            )
        except TimeoutError as exc:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            locked = blocked_hug_lock(
                left_now, right_now, open_left, open_right,
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
        if not hug_moved_from_pregrasp(reached_left, reached_right, open_left, open_right):
            result.setdefault("rejected_false_contacts", []).append({
                "clearance_m": plan["clearance_m"], "reason": "still_at_pregrasp",
            })
            continue
        if waypoint_result.get("blocked_hug"):
            tight = plans[-1]
            hold_left, hold_right = inward_hold_from_blocked(
                reached_left, reached_right, tight["left_joint_target"], tight["right_joint_target"],
            )
            result["blocked_hug_hold_blend"] = 0.22
            result["squeeze_skipped_blocked_hug"] = False
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
        squeeze = _confirm_inward_squeeze(
            node, hold_left, hold_right, args.gripper_open, args.squeeze_seconds, result,
        )
        contact_plan = plan
        contact_feedback = waypoint_result
        result["squeeze_confirmed"] = True
        result["squeeze_feedback"] = squeeze
        break
    if contact_plan is None:
        raise TimeoutError("dual-arm shelf contact was not established at any validated clearance waypoint")
    result["contact_detected"] = True
    result["contact_feedback"] = contact_feedback
    result["contact_clearance_detected_m"] = contact_plan["clearance_m"]
    result["reached_contact_plans"] = reached
    result["hold_joint_targets"] = {"left": hold_left.tolist(), "right": hold_right.tolist()}
    return {
        "hold_left": hold_left,
        "hold_right": hold_right,
        "lift_slide": lift_slide,
        "contact_slide": contact_slide,
        "blocked_hug": bool(contact_feedback.get("blocked_hug")),
        "held_center_base": box_base.copy(),
    }


def _spread_release_or_continue(node, left_start, right_start, left_end, right_end, args, result):
    """Open the hug; if settle fails, keep the last commanded pose and continue."""
    left_end = np.asarray(left_end, dtype=float)
    right_end = np.asarray(right_end, dtype=float)
    try:
        _traverse_pair(
            node, left_start, right_start, left_end, right_end,
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            args.gripper_open, args.gripper_open, True, result,
        )
        return left_end, right_end
    except TimeoutError as exc:
        result["release_spread_timeout"] = str(exc)
        node.controller.command_arm("l", left_end, args.gripper_open)
        node.controller.command_arm("r", right_end, args.gripper_open)
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            node.spin_once(0.05)
            _command_hug(node, left_end, right_end, args.gripper_open)
        left_now, right_now, _, _ = _current_arm_state_unbounded(node)
        result["release_spread_used_current"] = True
        return np.asarray(left_now, dtype=float), np.asarray(right_now, dtype=float)


def _finish_table_release(node, context, args, result):
    """Box is already on the table: spread, raise, reverse, then retract."""
    left_now, right_now, _, _ = _current_arm_state_unbounded(node)
    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    release_left = np.asarray(left_now, dtype=float)
    release_right = np.asarray(right_now, dtype=float)
    try:
        backend = MMK2KdlBackend()
        left_fk = backend.forward("l", current_slide, release_left)
        right_fk = backend.forward("r", current_slide, release_right)
        left_xyz, right_xyz = release_cartesian(left_fk[:3, 3], right_fk[:3, 3], args.release_spread)
        plan = solve_bimanual_pose(
            current_slide, release_left, release_right, left_xyz, right_xyz, backend=backend,
        )
        result["release_plan"] = plan
        release_left, release_right = _spread_release_or_continue(
            node, release_left, release_right,
            np.asarray(plan["left_joint_target"], dtype=float),
            np.asarray(plan["right_joint_target"], dtype=float),
            args, result,
        )
    except Exception as exc:
        result["release_plan_error"] = str(exc)
    context["release_left"] = release_left
    context["release_right"] = release_right
    open_plan = solve_bimanual_hug_pose(current_slide, release_left, release_right)
    context["high_left"] = np.asarray(open_plan["left_joint_target"], dtype=float)
    context["high_right"] = np.asarray(open_plan["right_joint_target"], dtype=float)
    _sleep_holding(node, 0.6, release_left, release_right, args.gripper_open, require_hold=False)
    result["released"] = True
    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    raise_slide = current_slide - TABLE_RELEASE_RAISE_M
    result["release_raise_slide"] = raise_slide
    _traverse_spine_keeping_pose(
        node, current_slide, raise_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        release_left, release_right, args.gripper_open, result,
    )
    result["release_raise_completed"] = True
    leave_start = _odom_pose(node)
    leave_target = pose_offset(leave_start, args.table_retreat, reverse=True)
    result["table_retreat_target"] = leave_target.tolist()
    retreat_final = _drive_line(
        node, leave_start, leave_target, -1, args.retreat_timeout, result,
        release_left, release_right, args.gripper_open,
        require_hold=False, min_traveled_m=0.16, position_tolerance=0.06,
        max_linear_speed=0.08, key_prefix="table_retreat",
    )
    result["table_retreat_final_base"] = retreat_final.tolist()
    result["table_retreat_completed"] = True
    _retract_arms(node, context, args, result)


def _recover(node, context, args, result, released: bool):
    try:
        node.controller.stop_base()
    except Exception:
        pass
    holding = context.get("hold_left") is not None and context.get("hold_right") is not None and not released
    placed = bool(result.get("place_lower_completed") and result.get("place_within_radius"))
    if placed:
        result["recovery_finish_after_place"] = True
        try:
            _finish_table_release(node, context, args, result)
        except Exception as exc:
            result["recovery_finish_error"] = str(exc)
            try:
                _retract_arms(node, context, args, result)
            except Exception as retract_exc:
                result["recovery_error"] = str(retract_exc)
        return
    if not holding and not released and context.get("open_left") is not None and context.get("open_right") is not None:
        try:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            if hug_moved_from_pregrasp(left_now, right_now, context["open_left"], context["open_right"]):
                tight = (result.get("contact_plans") or [None])[-1]
                if tight and tight.get("left_joint_target") is not None:
                    context["hold_left"], context["hold_right"] = inward_hold_from_blocked(
                        left_now, right_now, tight["left_joint_target"], tight["right_joint_target"],
                    )
                else:
                    context["hold_left"] = np.asarray(left_now, dtype=float)
                    context["hold_right"] = np.asarray(right_now, dtype=float)
                holding = True
                result["recovery_locked_current_hug"] = True
        except Exception as exc:
            result["recovery_lock_hug_error"] = str(exc)
    current = None
    try:
        current = _odom_pose(node)
    except Exception:
        current = None
    near_shelf = current is not None and float(current[0]) <= SHELF_ZONE_X_MAX
    if holding:
        result["recovery_kept_hold"] = True
        if near_shelf:
            try:
                _drive_line(
                    node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                    args.retreat_timeout, result, context["hold_left"], context["hold_right"], args.gripper_open,
                    require_hold=True, min_traveled_m=0.18, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                    key_prefix="recovery_shelf_retreat",
                    hold_keeper=_bind_hold_keeper(context, result),
                )
                result["recovery_held_retreat_completed"] = True
            except Exception as exc:
                result["recovery_held_retreat_error"] = str(exc)
        try:
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
        except Exception as exc:
            result["recovery_keep_hold_error"] = str(exc)
        return
    if (not holding) and near_shelf and current is not None:
        try:
            _drive_line(
                node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                args.retreat_timeout, result,
                context.get("open_left", context.get("initial_left")),
                context.get("open_right", context.get("initial_right")),
                args.gripper_open,
                require_hold=False, min_traveled_m=0.12, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                key_prefix="recovery_empty_retreat",
            )
        except Exception as exc:
            result["recovery_empty_retreat_error"] = str(exc)
    if released or not holding:
        try:
            _retract_arms(node, context, args, result)
        except Exception as exc:
            result["recovery_error"] = str(exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 2 shelf-to-table hug placement check")
    parser.add_argument("--box-world", nargs=3, type=float, help="optional L2 brown center override")
    parser.add_argument("--no-allow-fixed-brown", action="store_false", dest="allow_fixed_brown")
    parser.set_defaults(allow_fixed_brown=True)
    parser.add_argument("--place-world", nargs=3, type=float, default=INSTRUCTION_PLACE_WORLD.tolist())
    parser.add_argument("--place-radius", type=float, default=PLACE_RADIUS_M)
    parser.add_argument("--place-accept-radius", type=float, default=PLACE_ACCEPT_RADIUS_M)
    parser.add_argument("--place-yaw", type=float, default=PLACE_YAW)
    parser.add_argument("--pick-yaw", type=float, default=PICK_YAW)
    parser.add_argument("--initial-clearance", type=float, default=0.02)
    parser.add_argument("--contact-step", type=float, default=0.01)
    parser.add_argument("--grasp-fwd-offset", type=float, default=SHELF_GRASP_FWD_OFFSET_M)
    parser.add_argument("--grasp-z-offset", type=float, default=SHELF_GRASP_Z_OFFSET_M)
    parser.add_argument("--approach-half", type=float, default=SHELF_APPROACH_HALF_M)
    parser.add_argument("--hold-half", type=float, default=SHELF_HOLD_HALF_M)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-max-step", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=SHELF_LIFT_HEIGHT_M)
    parser.add_argument("--place-clearance", type=float, default=TABLE_PLACE_CLEARANCE_M)
    parser.add_argument("--release-spread", type=float, default=TABLE_RELEASE_SPREAD_M)
    parser.add_argument("--staging-back", type=float, default=STAGING_BACK_M)
    parser.add_argument("--shelf-retreat", type=float, default=SHELF_RETREAT_M)
    parser.add_argument("--table-retreat", type=float, default=TABLE_RETREAT_M)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--squeeze-seconds", type=float, default=DEFAULT_SQUEEZE_SECONDS)
    parser.add_argument("--nav-timeout", type=float, default=DEFAULT_NAV_TIMEOUT_SEC)
    parser.add_argument("--yaw-timeout", type=float, default=DEFAULT_YAW_TIMEOUT_SEC)
    parser.add_argument("--shelf-timeout", type=float, default=DEFAULT_SHELF_TIMEOUT_SEC)
    parser.add_argument("--retreat-timeout", type=float, default=DEFAULT_LINE_TIMEOUT_SEC)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task2_shelf_to_table_check.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.07 <= args.hold_half <= args.approach_half <= 0.11:
        parser.error("shelf hold-half must be within [0.07, approach-half], approach-half <= 0.11 m")
    if not 0.08 <= args.place_accept_radius <= args.place_radius:
        parser.error("place-accept-radius must be within [0.08, place-radius]")
    if not lifted_box_clears_upper_board(BROWN_FIXED_WORLD[2], args.lift_height):
        parser.error("lift height would push the brown box into the L3 board")
    place_world = validate_table_place_world(args.place_world)
    pick_yaw = wrap_to_pi(args.pick_yaw)
    place_yaw = wrap_to_pi(args.place_yaw)
    print(
        f"task2_shelf_to_table_check starting apply={bool(args.apply)} output={args.output}",
        flush=True,
    )

    result = {
        "mode": "task2_bimanual_hug_shelf_to_table_check",
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
        "place_world": place_world.tolist(),
        "place_radius_m": args.place_radius,
        "place_accept_radius_m": args.place_accept_radius,
        "place_yaw_rad": place_yaw,
        "pick_yaw_rad": pick_yaw,
        "place_clearance_m": args.place_clearance,
        "release_spread_m": args.release_spread,
        "staging_back_m": args.staging_back,
        "shelf_retreat_m": args.shelf_retreat,
        "table_retreat_m": args.table_retreat,
        "published_control_topics": [],
        "phase": "init",
        "pink_l3_must_stay": True,
    }
    node = None
    context = {
        "initial_left": None,
        "hold_left": None,
        "hold_right": None,
        "release_left": None,
        "release_right": None,
    }
    phase = "init"
    released = False
    command_issued = False
    try:
        node = Ros2MissionNode(node_name="task2_shelf_to_table_check")
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
        resume_release = bool(carrying and already_at_table_place(start_base))
        resume_leave = bool(carrying and in_shelf and not resume_release)
        resume_place = bool(carrying and not in_shelf and not resume_release)
        result["resumed_mid_pick"] = resume_leave
        result["resumed_table_place"] = resume_place
        result["resumed_table_release"] = resume_release
        if resume_leave:
            print("resuming in-shelf hug; skipping regrasp", flush=True)
        if resume_place:
            print("resuming held table place; skipping shelf pick", flush=True)
        if resume_release:
            print("resuming table release; box already on the table", flush=True)

        phase = "detect"
        result["phase"] = phase
        located = None
        if args.box_world is not None:
            center_world = validate_brown_world(args.box_world)
            located = {
                "center_world": center_world.tolist(),
                "center_base": _world_to_base(node, center_world).tolist(),
                "source": "cli",
            }
        elif resume_place or resume_release:
            located = {
                "center_world": BROWN_FIXED_WORLD.tolist(),
                "center_base": _world_to_base(node, BROWN_FIXED_WORLD).tolist(),
                "source": "resume_carry",
            }
        elif not args.apply:
            located = {
                "center_world": BROWN_FIXED_WORLD.tolist(),
                "center_base": _world_to_base(node, BROWN_FIXED_WORLD).tolist(),
                "source": "fixed_layout_dry_run",
            }
        else:
            _look_at_shelf(node, result)
            try:
                located = locate_brown(node, pick_yaw)
            except Exception as exc:
                result["vision_error"] = str(exc)
                if not args.allow_fixed_brown:
                    raise
                located = {
                    "center_world": BROWN_FIXED_WORLD.tolist(),
                    "center_base": _world_to_base(node, BROWN_FIXED_WORLD).tolist(),
                    "source": "fixed_layout_fallback",
                }
        print(f"phase={phase} detection_source={located.get('source')}", flush=True)
        box_world = snap_l2_box_center(located["center_world"])
        result["detection"] = located
        result["box_world_snapped"] = box_world.tolist()
        pick_stand = pick_stand_from_box(box_world, pick_yaw)
        stage_pose = staging_pose(pick_stand[:2], pick_yaw, args.staging_back)
        result["pick_stand"] = pick_stand.tolist()
        result["staging_pose"] = stage_pose.tolist()
        contact_slide = float(PRE_GRASP_Z0 - (box_world[2] + args.grasp_z_offset))
        if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError(f"contact slide {contact_slide:.4f} is outside {SLIDE_LIMITS}")
        result["contact_slide"] = contact_slide
        if not lifted_box_clears_upper_board(box_world[2], args.lift_height):
            raise RuntimeError("lift would collide with the occupied L3 board")

        if not args.apply:
            result["status"] = "dry_run"
            result["box_contact_commanded"] = False
            result["base_motion_commanded"] = False
            result["transport_or_place_commanded"] = False
            return 0

        command_issued = True
        keeper = _bind_hold_keeper(context, result)
        if resume_release:
            phase = "release"
            result["phase"] = phase
            result["place_lower_completed"] = True
            result["place_within_radius"] = True
            open_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
            context["initial_left"] = np.asarray(open_plan["left_joint_target"], dtype=float)
            context["initial_right"] = np.asarray(open_plan["right_joint_target"], dtype=float)
            context["initial_slide"] = min(float(initial_slide), 0.05)
            context["high_left"] = context["initial_left"]
            context["high_right"] = context["initial_right"]
            _finish_table_release(node, context, args, result)
            result["status"] = "passed"
            return 0
        hug = None
        if resume_leave or resume_place:
            phase = "resume_hold"
            result["phase"] = phase
            print(
                f"phase=resume_hold slide={initial_slide:.3f} resume_place={resume_place} "
                f"box_world={box_world.tolist()}",
                flush=True,
            )
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            hold_left, hold_right = local_carry_hold(initial_left, initial_right)
            try:
                box_base = held_center_from_palms(current_slide, initial_left, initial_right)
                result["held_center_source"] = "palm_fk"
            except Exception as exc:
                result["palm_center_error"] = str(exc)
                box_base = np.asarray(_world_to_base(node, box_world), dtype=float)
                result["held_center_source"] = "snapped_vision"
            result["box_base_at_hug"] = box_base.tolist()
            lift_slide = lift_slide_target(contact_slide, args.lift_height)
            hug = {
                "hold_left": hold_left,
                "hold_right": hold_right,
                "lift_slide": lift_slide,
            }
            context.update({
                "open_left": np.zeros(6),
                "open_right": np.zeros(6),
                "high_left": hold_left,
                "high_right": hold_right,
                "hold_left": hold_left,
                "hold_right": hold_right,
            })
            result["resume_hold_left"] = hold_left.tolist()
            result["resume_hold_right"] = hold_right.tolist()
            _command_hug(node, hold_left, hold_right, args.gripper_open)
            height_slide = current_slide
            if resume_leave and abs(current_slide - lift_slide) > 0.02:
                phase = "hug_lift"
                result["phase"] = phase
                _traverse_spine_holding(
                    node, current_slide, lift_slide, args.spine_max_step, args.settle_timeout,
                    args.spine_tolerance, hold_left, hold_right, args.gripper_open, result,
                    hold_keeper=keeper,
                )
                height_slide = lift_slide
            else:
                result["lift_already_at_target"] = True
            if result.get("held_center_source") == "palm_fk":
                context["held_center_base"] = apply_slide_keep_hold(box_base, current_slide, height_slide)
            else:
                context["held_center_base"] = apply_slide_keep_hold(box_base, contact_slide, height_slide)
            result["lift_completed"] = True
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            _sleep_holding(
                node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
                hold_keeper=keeper,
            )
            hug["hold_left"] = context["hold_left"]
            hug["hold_right"] = context["hold_right"]
        else:
            skip_staging = bool(already_on_pick_approach(start_base, pick_stand, stage_pose))
            result["skip_staging"] = skip_staging
            print(
                f"phase=apply skip_staging={skip_staging} pick_stand={pick_stand.tolist()} "
                f"box_world={box_world.tolist()}",
                flush=True,
            )
            saved_initial_base = list(result["initial_base"])
            if not skip_staging:
                phase = "staging_nav"
                result["phase"] = phase
                _navigate(
                    node, stage_pose, 0.08, 0.10, args.nav_timeout, MAX_STAGING_NAV_M,
                    0.12, 0.50, result,
                )
                result["empty_staging_navigation"] = {
                    "final_base": result.get("final_base"),
                    "remaining_position_error_m": result.get("remaining_position_error_m"),
                    "remaining_yaw_error_rad": result.get("remaining_yaw_error_rad"),
                    "navigation_phase": result.get("navigation_phase"),
                }
                result["staging_final_base"] = _odom_pose(node).tolist()
            else:
                phase = "face_west"
                result["phase"] = phase
                current = _odom_pose(node)
                _navigate(
                    node, np.array([current[0], current[1], pick_yaw], dtype=float),
                    0.08, 0.10, args.yaw_timeout, 0.40, 0.08, 0.50, result,
                )
                result["face_west_base"] = _odom_pose(node).tolist()
            result["initial_base"] = saved_initial_base

            phase = "shelf_pregrasp"
            result["phase"] = phase
            _traverse_spine_keeping_pose(
                node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
                initial_left, initial_right, args.gripper_open, result,
            )
            open_plan = solve_bimanual_hug_pose(contact_slide, initial_left, initial_right)
            result["open_pregrasp_plan"] = open_plan
            open_left, open_right = _traverse_pair(
                node, initial_left, initial_right, open_plan["left_joint_target"], open_plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD,
                initial_left_gripper, initial_right_gripper, True, result,
            )
            _traverse_grippers(
                node, open_left, open_right, initial_left_gripper, initial_right_gripper,
                args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result,
            )
            context["open_left"] = open_left
            context["open_right"] = open_right
            context["high_left"] = open_left
            context["high_right"] = open_right

            phase = "shelf_creep"
            result["phase"] = phase
            creep_start = _odom_pose(node)
            try:
                creep_final = _drive_line(
                    node, creep_start, pick_stand, 1, args.shelf_timeout, result,
                    open_left, open_right, args.gripper_open,
                    require_hold=False, position_tolerance=0.04, max_linear_speed=MAX_SHELF_LINEAR_SPEED,
                    key_prefix="shelf_creep",
                )
            except TimeoutError as exc:
                creep_final = _odom_pose(node)
                result["shelf_creep_timeout_error"] = str(exc)
            result["creep_final_base"] = creep_final.tolist()
            result["shelf_creep_completed"] = True
            box_base = np.asarray(_world_to_base(node, box_world), dtype=float)
            result["box_base_at_hug"] = box_base.tolist()
            if not (0.45 <= box_base[0] <= 0.78 and abs(box_base[1]) <= 0.12):
                raise RuntimeError(f"brown box is outside the hug window after creep: {box_base.tolist()}")

            phase = "hug_lift"
            result["phase"] = phase
            hug = _close_shelf_hug(node, args, box_base, open_left, open_right, contact_slide, result)
            context.update(hug)
            _traverse_spine_holding(
                node, contact_slide, hug["lift_slide"], args.spine_max_step, args.settle_timeout, args.spine_tolerance,
                context["hold_left"], context["hold_right"], args.gripper_open, result,
                hold_keeper=keeper,
            )
            context["held_center_base"] = apply_slide_keep_hold(box_base, contact_slide, hug["lift_slide"])
            result["lift_completed"] = True
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            _sleep_holding(
                node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
                hold_keeper=keeper,
            )

        if not resume_place:
            phase = "shelf_leave"
            result["phase"] = phase
            leave_start = _odom_pose(node)
            leave_target = pose_offset(leave_start, args.shelf_retreat, reverse=True)
            result["shelf_leave_target"] = leave_target.tolist()
            leave_final = _drive_line(
                node, leave_start, leave_target, -1, args.retreat_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, min_traveled_m=TABLE_LEAVE_MIN_TRAVELED_M,
                position_tolerance=0.06, max_linear_speed=MAX_SHELF_LINEAR_SPEED, key_prefix="shelf_leave",
                hold_keeper=keeper,
            )
            result["shelf_leave_final_base"] = leave_final.tolist()
            result["shelf_leave_completed"] = True
            if float(result.get("shelf_leave_traveled_m", 0.0)) < TABLE_LEAVE_MIN_TRAVELED_M:
                raise RuntimeError(
                    f"shelf leave traveled {result.get('shelf_leave_traveled_m')} m, "
                    f"need at least {TABLE_LEAVE_MIN_TRAVELED_M:.2f} m"
                )

        phase = "table_staging"
        result["phase"] = phase
        table_stand = place_stand_xy(place_world, place_yaw, table_place_held_center(context["held_center_base"]))
        table_stage = staging_pose(table_stand, place_yaw, args.staging_back)
        result["table_place_stand_xy"] = table_stand.tolist()
        result["table_staging_pose"] = table_stage.tolist()
        _face_yaw_holding(
            node, place_yaw, args.yaw_timeout, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            key_prefix="face_north",
            hold_keeper=keeper,
        )
        staging_final = _navigate_holding(
            node, table_stage, args.nav_timeout, MAX_STAGING_NAV_M, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        result["table_staging_final_base"] = staging_final.tolist()
        result["table_staging_completed"] = True

        phase = "table_raise"
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
        result["table_raise_completed"] = True

        phase = "table_approach"
        result["phase"] = phase
        approach_start = _odom_pose(node)
        table_stand = place_stand_xy(place_world, place_yaw, table_place_held_center(context["held_center_base"]))
        place_pose = np.array([table_stand[0], table_stand[1], place_yaw], dtype=float)
        result["table_place_stand_xy"] = table_stand.tolist()
        try:
            approach_final = _drive_line(
                node, approach_start, place_pose, 1, args.shelf_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, position_tolerance=0.03, max_linear_speed=0.08,
                key_prefix="table_approach",
                hold_keeper=keeper,
            )
        except TimeoutError as exc:
            approach_final = _odom_pose(node)
            result["table_approach_timeout_error"] = str(exc)
        result["table_approach_final_base"] = approach_final.tolist()
        result["table_approach_completed"] = True
        try:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            context["held_center_base"] = held_center_from_palms(
                float(node.sensors.joint_vector(["slide_joint"])[0]), left_now, right_now,
            )
            result["held_center_source_before_lower"] = "palm_fk"
        except Exception as exc:
            result["pre_lower_palm_center_error"] = str(exc)
        ready = box_inside_table_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
        )
        result["pre_lower_estimated_place_world"] = ready["held_world"]
        result["pre_lower_xy_error_m"] = ready["xy_error_m"]
        on_table = box_on_table_top(ready["held_world"])
        result["pre_lower_on_table"] = on_table
        if not on_table:
            phase = "table_creep"
            result["phase"] = phase
            creep_start = _odom_pose(node)
            table_stand = place_stand_xy(
                place_world, place_yaw, table_place_held_center(context["held_center_base"]),
            )
            creep_pose = np.array([table_stand[0], table_stand[1], place_yaw], dtype=float)
            result["table_creep_target"] = creep_pose.tolist()
            creep_final = _drive_line(
                node, creep_start, creep_pose, 1, args.shelf_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, position_tolerance=0.03, max_linear_speed=0.06,
                key_prefix="table_creep",
                hold_keeper=keeper,
            )
            result["table_creep_final_base"] = creep_final.tolist()
            try:
                left_now, right_now, _, _ = _current_arm_state_unbounded(node)
                context["held_center_base"] = held_center_from_palms(
                    float(node.sensors.joint_vector(["slide_joint"])[0]), left_now, right_now,
                )
            except Exception as exc:
                result["creep_palm_center_error"] = str(exc)
            ready = box_inside_table_radius(
                _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
            )
            result["pre_lower_estimated_place_world"] = ready["held_world"]
            result["pre_lower_xy_error_m"] = ready["xy_error_m"]
            on_table = box_on_table_top(ready["held_world"])
            result["pre_lower_on_table"] = on_table
        if not on_table or not ready["within_radius"]:
            raise RuntimeError(
                f"box is not on the table top (world={ready['held_world']}, "
                f"xy error {ready['xy_error_m']:.3f} m); refuse to lower"
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
        error = table_placement_error(estimated, place_world, args.place_radius)
        result["estimated_place_world"] = estimated.tolist()
        result["place_xy_error_m"] = error["xy_error_m"]
        result["place_z_error_m"] = error["z_error_m"]
        result["place_within_radius"] = error["within_radius"]
        result["place_lower_completed"] = True
        _sleep_holding(
            node, 1.0, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        if not error["within_radius"]:
            raise RuntimeError(
                f"estimated place xy error {error['xy_error_m']:.4f} m exceeds radius {args.place_radius:.2f} m"
            )

        phase = "release"
        result["phase"] = phase
        _finish_table_release(node, context, args, result)
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["phase"] = phase
        print(f"task2_shelf_to_table_check failed in {phase}: {exc}", flush=True)
        if command_issued and node is not None:
            _recover(node, context, args, result, released)
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
