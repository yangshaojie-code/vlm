"""Automated, non-contact Task 1 bimanual validation campaign.

This combines only reversible checks: progressive gripper opening while the
arms are high, the already validated lowered pre-contact motion, then bounded
inward moves that retain a positive clearance.  It never closes a gripper,
touches the box, transports the base, or enables the formal mission executor.
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
from task1_precontact_check import (
    DEFAULT_GRASP_FWD_OFFSET,
    DEFAULT_GRASP_Z_OFFSET,
    DEFAULT_TOP_TO_CENTER,
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _current_arm_state,
    _traverse_pair,
    _traverse_spine,
    load_position_reference,
    validate_approach_geometry,
)


# The fixed-layout pink RGB-D localization error is about 1 cm.  Keep at
# least 2 cm planned clearance in an unattended non-contact campaign.
MIN_NONCONTACT_CLEARANCE = 0.02
MAX_GRIPPER_STEP = 0.15


def clearance_schedule(start_clearance: float, final_clearance: float, step: float) -> list[float]:
    """Return a descending, inclusive positive-clearance schedule."""
    start_clearance = float(start_clearance)
    final_clearance = float(final_clearance)
    step = float(step)
    if not MIN_NONCONTACT_CLEARANCE <= final_clearance <= start_clearance <= 0.08:
        raise ValueError("clearances must satisfy 0.01 <= final <= start <= 0.08 m")
    if not 0.005 <= step <= 0.02:
        raise ValueError("clearance step must be within [0.005, 0.020] m")
    values = [start_clearance]
    current = start_clearance
    while current - final_clearance > 1e-9:
        current = max(final_clearance, current - step)
        values.append(round(current, 9))
    return values


def gripper_waypoints(initial_left: float, initial_right: float, target: float, max_step: float) -> list[tuple[float, float]]:
    """Build synchronized bounded gripper moves within the official [0, 1] range."""
    initial_left, initial_right, target, max_step = map(float, (initial_left, initial_right, target, max_step))
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (initial_left, initial_right, target)):
        raise ValueError("all gripper values must be finite and within [0.0, 1.0]")
    if not 0.02 <= max_step <= MAX_GRIPPER_STEP:
        raise ValueError(f"gripper max step must be within [0.02, {MAX_GRIPPER_STEP:.2f}]")
    count = max(1, int(math.ceil(max(abs(target - initial_left), abs(target - initial_right)) / max_step)))
    return [
        (
            initial_left + index / count * (target - initial_left),
            initial_right + index / count * (target - initial_right),
        )
        for index in range(1, count + 1)
    ]


def bounded_gripper_feedback(value: float) -> float:
    """Convert finite feedback to the controller's physical command range.

    Endpoint overshoot is a simulator measurement artifact.  It must be
    recorded by callers that need diagnostics, but cannot prevent a stopped
    base or a retreat sequence from completing.
    """
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("current gripper feedback is non-finite")
    return float(np.clip(value, 0.0, 1.0))


def _campaign_arm_state(node):
    left, right, left_gripper, right_gripper = _current_arm_state_unbounded(node)
    return left, right, bounded_gripper_feedback(left_gripper), bounded_gripper_feedback(right_gripper)


def _current_arm_state_unbounded(node):
    """Read finite arm feedback while deferring campaign-specific gripper bounds."""
    left_names = [f"left_arm_joint{i}" for i in range(1, 7)]
    right_names = [f"right_arm_joint{i}" for i in range(1, 7)]
    left = node.sensors.joint_vector(left_names)
    right = node.sensors.joint_vector(right_names)
    left_gripper = float(node.sensors.joint_vector(["left_arm_eef_gripper_joint"])[0])
    right_gripper = float(node.sensors.joint_vector(["right_arm_eef_gripper_joint"])[0])
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise RuntimeError("current arm feedback is invalid")
    return left, right, left_gripper, right_gripper


def command_gripper_value(value: float) -> float:
    """Convert tolerated feedback to a valid [0, 1] command endpoint."""
    return bounded_gripper_feedback(value)


def _register_pair_topics(result):
    result["published_control_topics"] = list(dict.fromkeys(
        result["published_control_topics"] + [
            "/left_arm_forward_position_controller/commands",
            "/right_arm_forward_position_controller/commands",
        ]
    ))


def _traverse_grippers(node, arm_left, arm_right, start_left, start_right, target_left, target_right, max_step, timeout, tolerance, result):
    """Move both grippers in feedback-checked increments while holding arm joints."""
    reached_left, reached_right = float(start_left), float(start_right)
    for left_gripper, right_gripper in gripper_waypoints(start_left, start_right, target_left, max_step):
        node.controller.command_arm("l", arm_left, left_gripper)
        node.controller.command_arm("r", arm_right, right_gripper)
        _register_pair_topics(result)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            node.spin_once(0.05)
            current_left, current_right, current_left_gripper, current_right_gripper = _campaign_arm_state(node)
            arm_error = max(
                float(np.max(np.abs(current_left - arm_left))),
                float(np.max(np.abs(current_right - arm_right))),
            )
            gripper_error = max(abs(current_left_gripper - left_gripper), abs(current_right_gripper - right_gripper))
            if arm_error <= tolerance and gripper_error <= tolerance:
                reached_left, reached_right = current_left_gripper, current_right_gripper
                break
        else:
            raise TimeoutError("gripper feedback did not reach a bounded waypoint")
    return reached_left, reached_right


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Automated non-contact Task 1 bimanual approach campaign")
    parser.add_argument("--position-report", required=True)
    parser.add_argument("--position-tolerance", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--initial-clearance", type=float, default=0.03)
    parser.add_argument("--final-clearance", type=float, default=0.02)
    parser.add_argument("--clearance-step", type=float, default=0.01)
    parser.add_argument("--grasp-fwd-offset", type=float, default=DEFAULT_GRASP_FWD_OFFSET)
    parser.add_argument("--grasp-z-offset", type=float, default=DEFAULT_GRASP_Z_OFFSET)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-max-step", type=float, default=0.10)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task1_bimanual_approach_campaign.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if args.settle_timeout <= 0.0 or args.spine_tolerance <= 0.0 or args.hold_seconds < 0.0:
        parser.error("timeouts/tolerance must be positive and hold-seconds non-negative")
    clearance_values = clearance_schedule(args.initial_clearance, args.final_clearance, args.clearance_step)
    if not 0.0 <= args.gripper_open <= 1.0:
        parser.error("gripper-open must be within [0.0, 1.0]")

    result = {
        "mode": "task1_bimanual_noncontact_approach_campaign",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "gripper_motion_commanded": bool(args.apply),
        "box_contact_commanded": False,
        "base_motion_commanded": False,
        "transport_or_place_commanded": False,
        "published_control_topics": [],
        "clearance_schedule_m": clearance_values,
        "gripper_open_target": args.gripper_open,
    }
    node = None
    initial_left = initial_right = high_left = high_right = None
    initial_left_gripper = initial_right_gripper = None
    initial_slide = target_slide = None
    arms_opened = spine_changed = command_issued = False
    try:
        node = Ros2MissionNode(node_name="task1_bimanual_approach_campaign")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_left, initial_right, initial_left_gripper, initial_right_gripper = _campaign_arm_state(node)
        initial_left_gripper_feedback = initial_left_gripper
        initial_right_gripper_feedback = initial_right_gripper
        initial_left_gripper = command_gripper_value(initial_left_gripper)
        initial_right_gripper = command_gripper_value(initial_right_gripper)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        if initial_slide > 0.10:
            raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
        located = load_position_reference(args.position_report, node, args.position_tolerance, args.yaw_tolerance)
        current_base = np.asarray(located["center_base"], dtype=float)
        target_z = float(current_base[2] + args.grasp_z_offset)
        target_slide = float(PRE_GRASP_Z0 - target_z)
        if not SLIDE_LIMITS[0] <= target_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError(f"computed approach slide {target_slide:.4f} is outside {SLIDE_LIMITS}")

        high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
        plans = []
        left_reference = np.asarray(high_plan["left_joint_target"])
        right_reference = np.asarray(high_plan["right_joint_target"])
        for clearance in clearance_values:
            left_target, right_target = validate_approach_geometry(
                current_base, clearance, args.grasp_fwd_offset, args.grasp_z_offset,
            )
            plan = solve_bimanual_pose(target_slide, left_reference, right_reference, left_target, right_target)
            plan["clearance_m"] = clearance
            plans.append(plan)
            left_reference = np.asarray(plan["left_joint_target"])
            right_reference = np.asarray(plan["right_joint_target"])
        result.update({
            "position_reference": located,
            "initial_slide": initial_slide,
            "target_slide": target_slide,
            "initial_left_gripper_feedback": initial_left_gripper_feedback,
            "initial_right_gripper_feedback": initial_right_gripper_feedback,
            "initial_left_gripper": initial_left_gripper,
            "initial_right_gripper": initial_right_gripper,
            "gripper_open_waypoint_count": len(gripper_waypoints(initial_left_gripper, initial_right_gripper, args.gripper_open, args.gripper_max_step)),
            "high_pregrasp_plan": high_plan,
            "clearance_plans": plans,
        })
        if not args.apply:
            result["gripper_motion_commanded"] = False
            result["status"] = "dry_run"
            return 0

        command_issued = True
        high_left, high_right = _traverse_pair(
            node, initial_left, initial_right,
            high_plan["left_joint_target"], high_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, 0.010,
            command_gripper_value(initial_left_gripper), command_gripper_value(initial_right_gripper), True, result,
        )
        opened_left, opened_right = _traverse_grippers(
            node, high_left, high_right,
            initial_left_gripper, initial_right_gripper,
            args.gripper_open, args.gripper_open,
            args.gripper_max_step, args.settle_timeout, 0.010, result,
        )
        result["reached_open_left_gripper"] = opened_left
        result["reached_open_right_gripper"] = opened_right
        spine_changed = True
        _traverse_spine(node, initial_slide, target_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        reached_left, reached_right = high_left, high_right
        reached_plans = []
        for plan in plans:
            reached_left, reached_right = _traverse_pair(
                node, reached_left, reached_right,
                plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, 0.010,
                args.gripper_open, args.gripper_open, True, result,
            )
            reached_plans.append({"clearance_m": plan["clearance_m"], "left": reached_left.tolist(), "right": reached_right.tolist()})
        result["reached_clearance_plans"] = reached_plans
        until = time.monotonic() + args.hold_seconds
        while time.monotonic() < until:
            node.spin_once(min(0.05, until - time.monotonic()))
        for plan in reversed(plans[:-1]):
            reached_left, reached_right = _traverse_pair(
                node, reached_left, reached_right,
                plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, 0.010,
                args.gripper_open, args.gripper_open, True, result,
            )
        high_left, high_right = _traverse_pair(
            node, reached_left, reached_right,
            high_plan["left_joint_target"], high_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, 0.010,
            args.gripper_open, args.gripper_open, True, result,
        )
        _traverse_spine(node, target_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
        home_left, home_right = _traverse_pair(
            node, high_left, high_right, initial_left, initial_right,
            args.joint_max_step, args.settle_timeout, 0.010,
            args.gripper_open, args.gripper_open, True, result,
        )
        restored_left, restored_right = _traverse_grippers(
            node, home_left, home_right,
            args.gripper_open, args.gripper_open,
            initial_left_gripper, initial_right_gripper,
            args.gripper_max_step, args.settle_timeout, 0.010, result,
        )
        result["returned_left_gripper"] = restored_left
        result["returned_right_gripper"] = restored_right
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if command_issued and node is not None and initial_left is not None and initial_right is not None:
            try:
                current_left, current_right, current_left_gripper, current_right_gripper = _campaign_arm_state(node)
                if high_left is not None and high_right is not None:
                    _traverse_pair(node, current_left, current_right, high_left, high_right, args.joint_max_step, args.settle_timeout, 0.015, command_gripper_value(current_left_gripper), command_gripper_value(current_right_gripper), True, result)
                    current_left, current_right = high_left, high_right
                if spine_changed:
                    current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
                    _traverse_spine(node, current_slide, initial_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance, True, result)
                _traverse_pair(node, current_left, current_right, initial_left, initial_right, args.joint_max_step, args.settle_timeout, 0.015, command_gripper_value(current_left_gripper), command_gripper_value(current_right_gripper), True, result)
                _traverse_grippers(node, initial_left, initial_right, current_left_gripper, current_right_gripper, initial_left_gripper, initial_right_gripper, args.gripper_max_step, args.settle_timeout, 0.015, result)
                result["recovery_published"] = True
            except Exception as restore_exc:
                result["recovery_error"] = str(restore_exc)
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
