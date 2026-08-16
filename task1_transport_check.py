"""Task 1 bounded bimanual-hug transport check with automatic return.

This is the first transport validation after a verified Task 1 contact/lift.
It lifts the pink box, reverses straight away from the table by 0.20 m with
odom feedback, returns to the exact pickup station, then replaces and retracts.
It never visits the shelf, releases the box elsewhere, or enables formal play.
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
from ros2_mission_node import Ros2MissionNode
from task1_bimanual_approach_campaign import (
    _current_arm_state_unbounded,
    _traverse_grippers,
    command_gripper_value,
)
from task1_pick_lift_check import (
    TASK1_APPROACH_HALF_M,
    TASK1_GRASP_FWD_OFFSET_M,
    TASK1_GRASP_Z_OFFSET_M,
    TASK1_HOLD_HALF_M,
    _approach_contact_pair,
    contact_approach_geometry,
    contact_clearance_schedule,
    contact_residuals,
    lift_slide_target,
)
from task1_precontact_check import (
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _traverse_pair,
    _traverse_spine,
    _yaw_from_odom,
    load_position_reference,
)


TRANSPORT_DISTANCE_M = 0.20
MAX_TRANSPORT_DISTANCE_M = 0.30
MAX_TRANSPORT_LINEAR_SPEED = 0.08
MAX_TRANSPORT_ANGULAR_SPEED = 0.30


def reverse_target(start_pose, distance: float) -> np.ndarray:
    """Return a same-yaw target behind the base along its current heading."""
    start = np.asarray(start_pose, dtype=float)
    distance = float(distance)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("start pose must be finite [x, y, yaw]")
    if not 0.20 <= distance <= MAX_TRANSPORT_DISTANCE_M:
        raise ValueError(f"transport distance must be within [0.20, {MAX_TRANSPORT_DISTANCE_M:.2f}] m")
    return np.array([
        start[0] - distance * math.cos(start[2]),
        start[1] - distance * math.sin(start[2]),
        start[2],
    ])


def transport_command(current_pose, start_pose, target_pose, direction: int, position_tolerance: float, yaw_tolerance: float):
    """Return a bounded straight-line transport command and diagnostic errors."""
    current = np.asarray(current_pose, dtype=float)
    start = np.asarray(start_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (reverse) or 1 (forward)")
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (current, start, target)):
        raise ValueError("all base poses must be finite [x, y, yaw]")
    heading = np.array([math.cos(start[2]), math.sin(start[2])])
    goal_delta = target[:2] - current[:2]
    remaining = float(np.linalg.norm(goal_delta))
    yaw_error = (start[2] - current[2] + math.pi) % (2.0 * math.pi) - math.pi
    traveled = float(abs(np.dot(current[:2] - start[:2], heading)))
    offset = current[:2] - start[:2]
    cross_track = float(abs(heading[0] * offset[1] - heading[1] * offset[0]))
    if remaining <= position_tolerance and abs(yaw_error) <= yaw_tolerance:
        return 0.0, 0.0, {"phase": "complete", "remaining_m": remaining, "traveled_m": traveled, "cross_track_m": cross_track, "yaw_error_rad": yaw_error}
    angular = float(np.clip(1.8 * yaw_error, -MAX_TRANSPORT_ANGULAR_SPEED, MAX_TRANSPORT_ANGULAR_SPEED))
    if abs(yaw_error) > 0.10:
        return 0.0, angular, {"phase": "align", "remaining_m": remaining, "traveled_m": traveled, "cross_track_m": cross_track, "yaw_error_rad": yaw_error}
    linear = direction * min(MAX_TRANSPORT_LINEAR_SPEED, max(0.03, 0.7 * remaining))
    return linear, angular, {"phase": "translate", "remaining_m": remaining, "traveled_m": traveled, "cross_track_m": cross_track, "yaw_error_rad": yaw_error}


def _odom_pose(node):
    odom = node.sensors.odom
    if odom is None:
        raise RuntimeError("waiting for odometry")
    pose = odom.pose.pose
    return np.array([pose.position.x, pose.position.y, _yaw_from_odom(odom)], dtype=float)


def _drive_held_box(node, start_pose, target_pose, direction, timeout, result, left_contact_target, right_contact_target):
    """Drive a short odom-closed-line segment while reasserting the hug pose."""
    deadline = time.monotonic() + float(timeout)
    final = None
    max_cross_track = 0.0
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        # The simulator occasionally reports a value just over 1.0 at the
        # gripper endpoint.  It is diagnostic feedback, not a reason to skip
        # the base stop/recovery path while holding a box.
        left_current, right_current, left_gripper, right_gripper = _current_arm_state_unbounded(node)
        samples = result.setdefault("transport_raw_gripper_feedback", [])
        if len(samples) < 20:
            samples.append({"left": float(left_gripper), "right": float(right_gripper)})
        if not (0.0 <= left_gripper <= 1.0 and 0.0 <= right_gripper <= 1.0):
            result["transport_gripper_endpoint_warning"] = True
        contact = contact_residuals(left_current, right_current, left_contact_target, right_contact_target)
        result["transport_contact_feedback"] = contact
        if not contact["symmetric_contact"]:
            node.controller.stop_base()
            raise RuntimeError(f"held-box contact changed during transport: {contact}")
        current = _odom_pose(node)
        linear, angular, details = transport_command(current, start_pose, target_pose, direction, 0.02, 0.05)
        max_cross_track = max(max_cross_track, details["cross_track_m"])
        final = current
        result.update({
            "transport_phase": details["phase"],
            "transport_remaining_m": details["remaining_m"],
            "transport_traveled_m": details["traveled_m"],
            "transport_cross_track_m": details["cross_track_m"],
            "transport_yaw_error_rad": details["yaw_error_rad"],
            "transport_max_cross_track_m": max_cross_track,
        })
        if details["phase"] == "complete":
            node.controller.stop_base()
            return final
        node.controller.publish_velocity(linear, angular)
        if "/cmd_vel" not in result["published_control_topics"]:
            result["published_control_topics"].append("/cmd_vel")
    node.controller.stop_base()
    raise TimeoutError(f"held-box transport timed out; final={None if final is None else final.tolist()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 1 bounded hug transport check with automatic return")
    parser.add_argument("--position-report", required=True)
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
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--transport-distance", type=float, default=TRANSPORT_DISTANCE_M)
    parser.add_argument("--transport-timeout", type=float, default=18.0)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task1_transport_check.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.0 <= args.gripper_open <= 1.0 or args.settle_timeout <= 0 or args.transport_timeout <= 0:
        parser.error("gripper or timeout arguments are invalid")
    if not TASK1_HOLD_HALF_M <= args.hold_half <= args.approach_half <= TASK1_APPROACH_HALF_M:
        parser.error("hold-half must be within [0.115, approach-half], and approach-half <= 0.13 m")
    clearances = contact_clearance_schedule(args.initial_clearance, args.contact_step)

    result = {
        "mode": "task1_bimanual_hug_short_transport_check",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "box_contact_commanded": bool(args.apply),
        "base_motion_commanded": bool(args.apply),
        "transport_or_place_commanded": False,
        "gripper_open_target": args.gripper_open,
        "approach_half_m": args.approach_half,
        "hold_half_m": args.hold_half,
        "grasp_fwd_offset_m": args.grasp_fwd_offset,
        "grasp_z_offset_m": args.grasp_z_offset,
        "contact_clearance_schedule_m": clearances,
        "lift_height_m": args.lift_height,
        "transport_distance_m": args.transport_distance,
        "published_control_topics": [],
    }
    node = None
    initial_left = initial_right = high_left = high_right = None
    initial_left_gripper = initial_right_gripper = None
    initial_slide = contact_slide = None
    start_base = outbound_target = None
    spine_changed = base_moved = command_issued = False
    try:
        node = Ros2MissionNode(node_name="task1_transport_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
        initial_left_gripper = command_gripper_value(raw_left_gripper)
        initial_right_gripper = command_gripper_value(raw_right_gripper)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        if initial_slide > 0.10:
            raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
        located = load_position_reference(args.position_report, node, args.position_tolerance, args.yaw_tolerance)
        box_base = np.asarray(located["center_base"], dtype=float)
        contact_slide = float(PRE_GRASP_Z0 - (box_base[2] + args.grasp_z_offset))
        if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError("contact slide target is outside limits")
        lift_slide = lift_slide_target(contact_slide, args.lift_height)
        high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
        plans = []
        left_reference, right_reference = np.asarray(high_plan["left_joint_target"]), np.asarray(high_plan["right_joint_target"])
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
        start_base = _odom_pose(node)
        outbound_target = reverse_target(start_base, args.transport_distance)
        result.update({
            "position_reference": located, "initial_slide": initial_slide,
            "initial_raw_gripper_feedback": {"left": raw_left_gripper, "right": raw_right_gripper},
            "contact_slide": contact_slide, "lift_slide": lift_slide,
            "initial_base": start_base.tolist(), "outbound_base_target": outbound_target.tolist(),
            "high_pregrasp_plan": high_plan, "contact_plans": plans,
        })
        if not args.apply:
            result["box_contact_commanded"] = False
            result["base_motion_commanded"] = False
            result["status"] = "dry_run"
            return 0

        command_issued = True
        high_left, high_right = _traverse_pair(node, initial_left, initial_right, high_plan["left_joint_target"], high_plan["right_joint_target"], args.joint_max_step, args.settle_timeout, 0.010, initial_left_gripper, initial_right_gripper, True, result)
        _traverse_grippers(node, high_left, high_right, initial_left_gripper, initial_right_gripper, args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result)
        spine_changed = True
        _traverse_spine(node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        reached_left, reached_right = high_left, high_right
        for plan in plans[:-1]:
            reached_left, reached_right = _traverse_pair(node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"], args.joint_max_step, args.settle_timeout, 0.010, args.gripper_open, args.gripper_open, True, result)
        contact_plan = plans[-1]
        reached_left, reached_right, contact_feedback = _approach_contact_pair(node, contact_plan["left_joint_target"], contact_plan["right_joint_target"], args.gripper_open, args.gripper_open, args.settle_timeout, 0.010)
        result["contact_detected"] = True
        result["contact_feedback"] = contact_feedback
        _traverse_spine(node, contact_slide, lift_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        result["lift_completed"] = True
        base_moved = True
        outbound_final = _drive_held_box(
            node, start_base, outbound_target, -1, args.transport_timeout, result,
            contact_plan["left_joint_target"], contact_plan["right_joint_target"],
        )
        result["outbound_final_base"] = outbound_final.tolist()
        result["outbound_completed"] = True
        return_final = _drive_held_box(
            node, outbound_target, start_base, 1, args.transport_timeout, result,
            contact_plan["left_joint_target"], contact_plan["right_joint_target"],
        )
        result["return_final_base"] = return_final.tolist()
        result["return_completed"] = True
        _traverse_spine(node, lift_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        for plan in reversed(plans[:-1]):
            reached_left, reached_right = _traverse_pair(node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"], args.joint_max_step, args.settle_timeout, 0.010, args.gripper_open, args.gripper_open, True, result)
        high_left, high_right = _traverse_pair(node, reached_left, reached_right, high_plan["left_joint_target"], high_plan["right_joint_target"], args.joint_max_step, args.settle_timeout, 0.010, args.gripper_open, args.gripper_open, True, result)
        _traverse_spine(node, contact_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        home_left, home_right = _traverse_pair(node, high_left, high_right, initial_left, initial_right, args.joint_max_step, args.settle_timeout, 0.010, args.gripper_open, args.gripper_open, True, result)
        _traverse_grippers(node, home_left, home_right, args.gripper_open, args.gripper_open, initial_left_gripper, initial_right_gripper, args.gripper_max_step, args.settle_timeout, 0.010, result)
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if node is not None:
            try:
                node.controller.stop_base()
            except Exception:
                pass
        # Return to pickup station before lowering a potentially held box.
        if base_moved and node is not None and start_base is not None:
            try:
                current = _odom_pose(node)
                _drive_held_box(
                    node, current, start_base, 1, args.transport_timeout, result,
                    contact_plan["left_joint_target"], contact_plan["right_joint_target"],
                )
                result["return_after_failure_completed"] = True
            except Exception as return_exc:
                result["return_after_failure_error"] = str(return_exc)
        if command_issued and node is not None and initial_left is not None:
            try:
                current_left, current_right, left_gripper, right_gripper = _current_arm_state_unbounded(node)
                result["recovery_raw_gripper_feedback"] = {
                    "left": float(left_gripper), "right": float(right_gripper),
                }
                if high_left is not None and high_right is not None:
                    _traverse_pair(node, current_left, current_right, high_left, high_right, args.joint_max_step, args.settle_timeout, 0.015, float(np.clip(left_gripper, 0.0, 1.0)), float(np.clip(right_gripper, 0.0, 1.0)), True, result)
                    current_left, current_right = high_left, high_right
                if spine_changed:
                    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
                    _traverse_spine(node, current_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
                _traverse_pair(node, current_left, current_right, initial_left, initial_right, args.joint_max_step, args.settle_timeout, 0.015, float(np.clip(left_gripper, 0.0, 1.0)), float(np.clip(right_gripper, 0.0, 1.0)), True, result)
            except Exception as recovery_exc:
                result["recovery_error"] = str(recovery_exc)
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
