"""First Task 1 bimanual hug contact and lift check with automatic recovery.

This is intentionally a validation stop-point, not formal task execution. It
uses both open grippers and a bounded lateral hug to contact the pink box,
lifts only by the slide, returns it to the original table height, and retracts.
It never commands the base, shelf navigation, placement, or the referee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0, solve_bimanual_hug_pose, solve_bimanual_pose
from head_camera_kinematics import SLIDE_LIMITS
from ros2_mission_node import Ros2MissionNode
from task1_bimanual_approach_campaign import (
    _campaign_arm_state,
    _current_arm_state_unbounded,
    _traverse_grippers,
    command_gripper_value,
)
from task1_precontact_check import (
    DEFAULT_GRIP_HALF,
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _traverse_pair,
    _traverse_spine,
    load_position_reference,
)


MAX_TEST_LIFT_M = 0.12
CONTACT_MIN_JOINT_RESIDUAL_RAD = 0.015
CONTACT_MAX_JOINT_RESIDUAL_RAD = 0.08
# Empty close settles near 0.006 rad. An offset box can leave one arm just
# under CONTACT_MIN while the other is still blocked; that is still a hold.
CARRY_EMPTY_JOINT_RESIDUAL_RAD = 0.010
CONTACT_STABLE_SAMPLES = 3
# Real Server feedback settled within about 0.01 rad at a non-contact
# approach waypoint. Keep a small margin here; the final contact check below
# still requires simultaneous bounded residuals on both arms.
APPROACH_JOINT_TOLERANCE_RAD = 0.015
# A real hug from the high pre-grasp pose moves joint 2 by about 0.9 rad.
# Anything much smaller is still the raised pose, even if a tiny interpolation
# step happens to fall in the contact residual band.
MIN_HUG_JOINT_DELTA_FROM_PREGRASP_RAD = 0.30
# Once the box has been lowered back onto the table, the arms can be
# mechanically displaced by contact/friction while opening out.  This is a
# recovery/retraction phase, not evidence of failed contact, so use a bounded
# but more forgiving feedback tolerance there.
RETRACTION_JOINT_TOLERANCE_RAD = 0.050
# These are the verified fixed-layout Task 1 values from the supplied
# reference client.  The wider 0.13 m pose is only an approach waypoint;
# carrying uses a slightly tighter, but not over-compressed, 0.115 m half gap.
TASK1_APPROACH_HALF_M = 0.13
TASK1_HOLD_HALF_M = 0.115
TASK1_GRASP_FWD_OFFSET_M = 0.065
TASK1_GRASP_Z_OFFSET_M = 0.045


def contact_approach_geometry(box_base, clearance: float, grasp_fwd_offset: float, grasp_z_offset: float, hold_half: float = DEFAULT_GRIP_HALF):
    """Build the Task 1 hug targets including exactly 0 m planned lateral clearance."""
    box_base = np.asarray(box_base, dtype=float)
    clearance = float(clearance)
    if box_base.shape != (3,) or not np.all(np.isfinite(box_base)):
        raise ValueError("box_base must contain three finite values")
    if not 0.0 <= clearance <= 0.05:
        raise ValueError("contact clearance must be within [0.0, 0.05] m")
    if not 0.0 <= float(grasp_fwd_offset) <= 0.10 or not 0.0 <= float(grasp_z_offset) <= 0.10:
        raise ValueError("grasp offsets are outside the validated bounds")
    if not 0.10 <= float(hold_half) <= 0.13:
        raise ValueError("hold half gap must be within [0.10, 0.13] m")
    if not 0.30 <= box_base[0] <= 0.80 or abs(box_base[1]) > 0.18:
        raise ValueError(f"box is outside the safe base-frame approach window: {box_base.tolist()}")
    center = box_base + np.array([float(grasp_fwd_offset), 0.0, float(grasp_z_offset)])
    half = float(hold_half) + clearance
    return center + np.array([0.0, half, 0.0]), center + np.array([0.0, -half, 0.0])


def contact_clearance_schedule(start_clearance: float, step: float) -> list[float]:
    """Return a descending schedule ending at physical lateral contact (0 m)."""
    start_clearance = float(start_clearance)
    step = float(step)
    if not 0.02 <= start_clearance <= 0.05:
        raise ValueError("initial clearance must be within [0.02, 0.05] m")
    if not 0.005 <= step <= 0.01:
        raise ValueError("contact step must be within [0.005, 0.010] m")
    values = [start_clearance]
    current = start_clearance
    while current > 1e-9:
        current = max(0.0, current - step)
        values.append(round(current, 9))
    return values


def lift_slide_target(contact_slide: float, lift_height: float) -> float:
    """Decreasing slide position raises the fixed arm/end-effector geometry."""
    contact_slide = float(contact_slide)
    lift_height = float(lift_height)
    if not 0.05 <= lift_height <= MAX_TEST_LIFT_M:
        raise ValueError(f"lift height must be within [0.05, {MAX_TEST_LIFT_M:.2f}] m")
    target = contact_slide - lift_height
    if not SLIDE_LIMITS[0] <= target <= SLIDE_LIMITS[1]:
        raise ValueError(f"lift slide target {target:.4f} is outside {SLIDE_LIMITS}")
    return target


def contact_residuals(left_current, right_current, left_target, right_target, tolerance: float = 0.010):
    """Classify a symmetric, bounded residual as physical dual-arm contact.

    The final hug waypoint intentionally asks both arms to occupy the box
    boundary.  A simultaneously bounded residual on both arms is therefore a
    valid contact event, while a one-sided or excessive residual is unsafe and
    remains a failure.
    """
    vectors = [np.asarray(value, dtype=float) for value in (left_current, right_current, left_target, right_target)]
    if any(vector.shape != (6,) or not np.all(np.isfinite(vector)) for vector in vectors):
        raise ValueError("contact vectors must be finite six-joint arrays")
    left_error = float(np.max(np.abs(vectors[0] - vectors[2])))
    right_error = float(np.max(np.abs(vectors[1] - vectors[3])))
    target_reached = left_error <= tolerance and right_error <= tolerance
    symmetric_contact = (
        CONTACT_MIN_JOINT_RESIDUAL_RAD <= left_error <= CONTACT_MAX_JOINT_RESIDUAL_RAD
        and CONTACT_MIN_JOINT_RESIDUAL_RAD <= right_error <= CONTACT_MAX_JOINT_RESIDUAL_RAD
    )
    return {
        "left_max_joint_residual_rad": left_error,
        "right_max_joint_residual_rad": right_error,
        "target_reached": target_reached,
        "symmetric_contact": symmetric_contact,
    }


def holding_pose_ok(left_current, right_current, left_hold, right_hold):
    """True while both arms remain near the locked carry pose.

    After lift the controllers may settle onto the hug command, so a residual
    below CONTACT_MIN is still a valid hold.  The hug is lost only when either
    arm leaves the carry pose by more than CONTACT_MAX.
    """
    residual = contact_residuals(left_current, right_current, left_hold, right_hold)
    residual["holding"] = (
        residual["left_max_joint_residual_rad"] <= CONTACT_MAX_JOINT_RESIDUAL_RAD
        and residual["right_max_joint_residual_rad"] <= CONTACT_MAX_JOINT_RESIDUAL_RAD
    )
    return residual


def carry_squeeze_ok(left_current, right_current, left_squeeze, right_squeeze):
    """True only while the box still blocks the inward carry setpoint.

    Commanding a slightly tighter pose than the first-touch joints is what
    produces holding force.  If both arms then settle onto that setpoint, the
    box is no longer between them.
    """
    residual = contact_residuals(left_current, right_current, left_squeeze, right_squeeze)
    residual["holding"] = residual["symmetric_contact"]
    return residual


def carry_hold_ok(left_current, right_current, left_hold, right_hold):
    """True while the box still spans the hug, including a slight left/right offset.

    Confirming the squeeze still requires symmetric_contact.  After that, a
    rotated or laterally offset box can block one arm more than the other.
    The hold is lost when an arm leaves the pose, either palm closes onto the
    empty setpoint, or neither arm is blocked.
    """
    residual = contact_residuals(left_current, right_current, left_hold, right_hold)
    left = residual["left_max_joint_residual_rad"]
    right = residual["right_max_joint_residual_rad"]
    residual["holding"] = (
        left <= CONTACT_MAX_JOINT_RESIDUAL_RAD
        and right <= CONTACT_MAX_JOINT_RESIDUAL_RAD
        and max(left, right) >= CONTACT_MIN_JOINT_RESIDUAL_RAD
        and min(left, right) >= CARRY_EMPTY_JOINT_RESIDUAL_RAD
    )
    return residual


def classify_approach_sample(
    left_current,
    right_current,
    left_waypoint,
    right_waypoint,
    left_plan_target,
    right_plan_target,
    settle_tolerance: float,
):
    """Separate interpolation-step lag from contact against the clearance pose.

    The first 0.05 rad hug command can lag by about 0.05 rad on both arms while
    they are still at the high pre-grasp pose.  That lag sits inside the contact
    residual band if it is compared with the interpolation waypoint.  Contact is
    therefore judged only against this clearance's final joint targets.
    """
    waypoint = contact_residuals(
        left_current, right_current, left_waypoint, right_waypoint, settle_tolerance,
    )
    plan = contact_residuals(
        left_current, right_current, left_plan_target, right_plan_target, settle_tolerance,
    )
    return {
        "waypoint_reached": waypoint["target_reached"],
        "plan_reached": plan["target_reached"],
        "plan_contact": plan["symmetric_contact"],
        "left_max_joint_residual_rad": plan["left_max_joint_residual_rad"],
        "right_max_joint_residual_rad": plan["right_max_joint_residual_rad"],
        "waypoint_left_max_joint_residual_rad": waypoint["left_max_joint_residual_rad"],
        "waypoint_right_max_joint_residual_rad": waypoint["right_max_joint_residual_rad"],
        "target_reached": waypoint["target_reached"],
        "symmetric_contact": plan["symmetric_contact"],
    }


def hug_moved_from_pregrasp(left_current, right_current, left_pregrasp, right_pregrasp) -> bool:
    """True when both arms have left the raised pre-grasp pose far enough to hug."""
    left_delta = float(np.max(np.abs(np.asarray(left_current, dtype=float) - np.asarray(left_pregrasp, dtype=float))))
    right_delta = float(np.max(np.abs(np.asarray(right_current, dtype=float) - np.asarray(right_pregrasp, dtype=float))))
    return (
        left_delta >= MIN_HUG_JOINT_DELTA_FROM_PREGRASP_RAD
        and right_delta >= MIN_HUG_JOINT_DELTA_FROM_PREGRASP_RAD
    )


def _approach_until_reached_or_contact(
    node, left_start, right_start, left_target, right_target,
    max_step, left_gripper, right_gripper, timeout, tolerance, result,
    allow_early_contact: bool = True,
):
    """Advance one hug target, stopping on convergence or stable dual contact.

    The box dimensions reported by vision can differ slightly from the collision
    mesh.  Therefore a genuine symmetric contact can occur at an earlier
    clearance waypoint.  Continuing to demand the next inward waypoint would
    press into the box and, more importantly, turn a valid grasp into a false
    controller-settling failure.

    Contact is judged against this clearance's final joint targets, never against
    intermediate interpolation waypoints.  The 0.02 m approach pose still has a
    physical gap, so callers can disable early contact there.
    """
    left_start, right_start = np.asarray(left_start), np.asarray(right_start)
    left_target, right_target = np.asarray(left_target), np.asarray(right_target)
    count = max(1, int(np.ceil(max(
        float(np.max(np.abs(left_target - left_start))),
        float(np.max(np.abs(right_target - right_start))),
    ) / float(max_step))))
    result.setdefault("waypoint_counts", []).append(count)
    left_current, right_current = left_start, right_start
    last = None
    for index in range(1, count + 1):
        fraction = index / count
        left_waypoint = left_start + fraction * (left_target - left_start)
        right_waypoint = right_start + fraction * (right_target - right_start)
        node.controller.command_arm("l", left_waypoint, left_gripper)
        node.controller.command_arm("r", right_waypoint, right_gripper)
        deadline = time.monotonic() + float(timeout)
        stable_contact = 0
        while time.monotonic() < deadline:
            node.spin_once(0.05)
            left_current, right_current, _left_gripper, _right_gripper = _campaign_arm_state(node)
            last = classify_approach_sample(
                left_current, right_current, left_waypoint, right_waypoint,
                left_target, right_target, tolerance,
            )
            if allow_early_contact and last["plan_contact"]:
                stable_contact += 1
                if stable_contact >= CONTACT_STABLE_SAMPLES:
                    last["contact_detected"] = True
                    return left_current, right_current, last
            else:
                stable_contact = 0
            if last["waypoint_reached"]:
                break
        else:
            if allow_early_contact and last is not None and last["plan_contact"]:
                last["contact_detected"] = True
                return left_current, right_current, last
            raise TimeoutError(
                "paired arms did not settle; "
                f"left={left_current}, right={right_current}; last={last}"
            )
    last["contact_detected"] = False
    return left_current, right_current, last


def _approach_contact_pair(node, left_target, right_target, left_gripper, right_gripper, timeout, tolerance):
    """Compatibility helper for callers that intentionally target final contact.

    The transport check owns its bounded waypoint traversal before calling this
    final-contact helper.  Keep that contract while the lift-only check can
    detect contact at an earlier safe clearance waypoint.
    """
    node.controller.command_arm("l", left_target, left_gripper)
    node.controller.command_arm("r", right_target, right_gripper)
    deadline = time.monotonic() + float(timeout)
    stable_contact = 0
    last = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, _left_gripper, _right_gripper = _campaign_arm_state(node)
        last = contact_residuals(left_current, right_current, left_target, right_target, tolerance)
        if last["symmetric_contact"]:
            stable_contact += 1
            if stable_contact >= CONTACT_STABLE_SAMPLES:
                return left_current, right_current, last
        else:
            stable_contact = 0
    raise TimeoutError(f"dual-arm contact was not established safely; last={last}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 1 bounded contact-and-lift hug check")
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
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--approach-joint-tolerance", type=float, default=APPROACH_JOINT_TOLERANCE_RAD)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task1_pick_lift_check.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if (
        not 0.0 <= args.gripper_open <= 1.0
        or args.settle_timeout <= 0
        or args.spine_tolerance <= 0
        or not 0.010 <= args.approach_joint_tolerance <= 0.020
        or args.hold_seconds < 0
    ):
        parser.error("gripper, timeout, tolerance, or hold arguments are invalid")
    if not TASK1_HOLD_HALF_M <= args.hold_half <= args.approach_half <= TASK1_APPROACH_HALF_M:
        parser.error("hold-half must be within [0.115, approach-half], and approach-half <= 0.13 m")
    clearances = contact_clearance_schedule(args.initial_clearance, args.contact_step)

    result = {
        "mode": "task1_bimanual_hug_contact_lift_check",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "gripper_motion_commanded": bool(args.apply),
        "box_contact_commanded": bool(args.apply),
        "planned_contact_clearance_m": 0.0,
        "approach_half_m": args.approach_half,
        "hold_half_m": args.hold_half,
        "grasp_fwd_offset_m": args.grasp_fwd_offset,
        "grasp_z_offset_m": args.grasp_z_offset,
        "base_motion_commanded": False,
        "transport_or_place_commanded": False,
        "contact_clearance_schedule_m": clearances,
        "lift_height_m": args.lift_height,
        "approach_joint_tolerance_rad": args.approach_joint_tolerance,
        "published_control_topics": [],
        "phase": "initializing",
    }
    node = None
    initial_left = initial_right = high_left = high_right = None
    initial_left_gripper = initial_right_gripper = None
    initial_slide = contact_slide = None
    spine_changed = command_issued = False
    try:
        node = Ros2MissionNode(node_name="task1_pick_lift_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
        initial_left_gripper = command_gripper_value(raw_left_gripper)
        initial_right_gripper = command_gripper_value(raw_right_gripper)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        if initial_slide > 0.10:
            raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
        located = load_position_reference(args.position_report, node, args.position_tolerance, args.yaw_tolerance)
        box_base = np.asarray(located["center_base"], dtype=float)
        contact_z = float(box_base[2] + args.grasp_z_offset)
        contact_slide = float(PRE_GRASP_Z0 - contact_z)
        lift_slide = lift_slide_target(contact_slide, args.lift_height)
        high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)

        plans = []
        left_reference = np.asarray(high_plan["left_joint_target"])
        right_reference = np.asarray(high_plan["right_joint_target"])
        for clearance in clearances:
            # Keep the reference client's 0.13 m lateral geometry while
            # approaching, then use its 0.115 m stable carrying geometry.
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
            "position_reference": located,
            "initial_slide": initial_slide,
            "contact_slide": contact_slide,
            "lift_slide": lift_slide,
            "initial_left_gripper": initial_left_gripper,
            "initial_right_gripper": initial_right_gripper,
            "initial_raw_gripper_feedback": {"left": raw_left_gripper, "right": raw_right_gripper},
            "high_pregrasp_plan": high_plan,
            "contact_plans": plans,
        })
        if not args.apply:
            result["gripper_motion_commanded"] = False
            result["box_contact_commanded"] = False
            result["status"] = "dry_run"
            return 0

        command_issued = True
        result["phase"] = "pregrasp"
        high_left, high_right = _traverse_pair(
            node, initial_left, initial_right, high_plan["left_joint_target"], high_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, args.approach_joint_tolerance,
            initial_left_gripper, initial_right_gripper, True, result,
        )
        open_left, open_right = _traverse_grippers(
            node, high_left, high_right, initial_left_gripper, initial_right_gripper,
            args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result,
        )
        result["reached_open_left_gripper"] = open_left
        result["reached_open_right_gripper"] = open_right
        spine_changed = True
        result["phase"] = "lowering_to_contact"
        _traverse_spine(node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        reached_left, reached_right = high_left, high_right
        reached = []
        contact_plan = None
        contact_plan_index = None
        contact_result = None
        for plan_index, plan in enumerate(plans):
            result["phase"] = f"approach_clearance_{plan['clearance_m']:.3f}"
            result["published_control_topics"] = list(dict.fromkeys(result["published_control_topics"] + [
                "/left_arm_forward_position_controller/commands",
                "/right_arm_forward_position_controller/commands",
            ]))
            if plan["clearance_m"] > 1e-9:
                reached_left, reached_right = _traverse_pair(
                    node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                    args.joint_max_step, args.settle_timeout, args.approach_joint_tolerance,
                    args.gripper_open, args.gripper_open, True, result,
                )
                waypoint_result = {"contact_detected": False}
            else:
                reached_left, reached_right, waypoint_result = _approach_until_reached_or_contact(
                    node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                    args.joint_max_step, args.gripper_open, args.gripper_open,
                    args.settle_timeout, args.approach_joint_tolerance, result,
                )
            reached.append({
                "clearance_m": plan["clearance_m"],
                "left": reached_left.tolist(),
                "right": reached_right.tolist(),
                "contact_detected": waypoint_result["contact_detected"],
            })
            if waypoint_result["contact_detected"]:
                if not hug_moved_from_pregrasp(reached_left, reached_right, high_left, high_right):
                    raise RuntimeError(
                        "contact was reported while both arms were still at the high pre-grasp pose"
                    )
                contact_plan = plan
                contact_plan_index = plan_index
                contact_result = waypoint_result
                break
        if contact_plan is None:
            raise TimeoutError("dual-arm contact was not established at any validated clearance waypoint")
        result["contact_detected"] = True
        result["contact_feedback"] = contact_result
        result["contact_clearance_detected_m"] = contact_plan["clearance_m"]
        result["reached_contact_plans"] = reached
        result["phase"] = "contact_established"
        result["contact_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0])
        result["phase"] = "lifting"
        _traverse_spine(node, contact_slide, lift_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        result["lift_completed"] = True
        result["lift_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0])
        result["phase"] = "holding"
        until = time.monotonic() + args.hold_seconds
        while time.monotonic() < until:
            node.spin_once(min(0.05, until - time.monotonic()))
        result["phase"] = "lowering_after_hold"
        _traverse_spine(node, lift_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        result["lowered_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0])
        for plan in reversed(plans[:contact_plan_index]):
            result["phase"] = f"retracting_clearance_{plan['clearance_m']:.3f}"
            reached_left, reached_right = _traverse_pair(
                node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
                args.gripper_open, args.gripper_open, True, result,
            )
        high_left, high_right = _traverse_pair(
            node, reached_left, reached_right, high_plan["left_joint_target"], high_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            args.gripper_open, args.gripper_open, True, result,
        )
        _traverse_spine(node, contact_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        home_left, home_right = _traverse_pair(
            node, high_left, high_right, initial_left, initial_right,
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            args.gripper_open, args.gripper_open, True, result,
        )
        returned_left, returned_right = _traverse_grippers(
            node, home_left, home_right, args.gripper_open, args.gripper_open,
            initial_left_gripper, initial_right_gripper, args.gripper_max_step, args.settle_timeout, 0.010, result,
        )
        result["returned_left_gripper"] = returned_left
        result["returned_right_gripper"] = returned_right
        result["phase"] = "complete"
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result.setdefault("failure_phase", result.get("phase"))
        try:
            result["failure_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0]) if node is not None else None
        except Exception:
            pass
        if command_issued and node is not None and initial_left is not None:
            try:
                current_left, current_right, left_gripper, right_gripper = _current_arm_state_unbounded(node)
                if high_left is not None and high_right is not None:
                    _traverse_pair(node, current_left, current_right, high_left, high_right, args.joint_max_step, args.settle_timeout, 0.015, command_gripper_value(left_gripper), command_gripper_value(right_gripper), True, result)
                    current_left, current_right = high_left, high_right
                if spine_changed:
                    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
                    _traverse_spine(node, current_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
                _traverse_pair(node, current_left, current_right, initial_left, initial_right, args.joint_max_step, args.settle_timeout, 0.015, command_gripper_value(left_gripper), command_gripper_value(right_gripper), True, result)
                _traverse_grippers(node, initial_left, initial_right, command_gripper_value(left_gripper), command_gripper_value(right_gripper), initial_left_gripper, initial_right_gripper, args.gripper_max_step, args.settle_timeout, 0.015, result)
                result["recovery_published"] = True
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
