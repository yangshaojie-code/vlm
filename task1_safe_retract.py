"""Recover a Task 1 hug check that stopped while holding the pink box.

This tool is deliberately narrower than the integration checks.  It never
commands the base or formal executor: it first lowers a lifted box to the
known table contact height, retracts both arms laterally, then returns the
slide and arm joints to a high neutral posture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0, solve_bimanual_hug_pose, solve_bimanual_pose
from ros2_mission_node import Ros2MissionNode
from task1_bimanual_approach_campaign import _current_arm_state_unbounded
from task1_pick_lift_check import TASK1_GRASP_Z_OFFSET_M
from task1_precontact_check import (
    MAX_ARM_STEP,
    MAX_SPINE_STEP,
    _traverse_pair,
    _traverse_spine,
    load_position_reference,
)

LEGACY_GRASP_FWD_OFFSET_M = -0.05
LEGACY_HOLD_HALF_M = 0.13


def recovery_slide_target(current_slide: float, contact_slide: float) -> float:
    """Only lower a possibly lifted box; never raise it before arm retraction."""
    current_slide = float(current_slide)
    contact_slide = float(contact_slide)
    if not np.isfinite(current_slide) or not np.isfinite(contact_slide):
        raise ValueError("slide values must be finite")
    return max(current_slide, contact_slide)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Safely retract a stopped Task 1 hug posture")
    parser.add_argument("--position-report", required=True)
    parser.add_argument("--position-tolerance", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--home-slide", type=float, default=0.03)
    parser.add_argument("--current-grasp-fwd-offset", type=float, default=LEGACY_GRASP_FWD_OFFSET_M)
    parser.add_argument("--current-hold-half", type=float, default=LEGACY_HOLD_HALF_M)
    parser.add_argument("--lateral-release-margin", type=float, default=0.05)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task1_safe_retract.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.0 <= args.home_slide <= 0.10 or args.settle_timeout <= 0.0:
        parser.error("home slide or timeout is invalid")
    if not -0.05 <= args.current_grasp_fwd_offset <= 0.065 or not 0.10 <= args.current_hold_half <= 0.13:
        parser.error("current hug geometry is outside the verified range")
    if not 0.03 <= args.lateral_release_margin <= 0.06:
        parser.error("lateral release margin must be within [0.03, 0.06] m")

    result = {
        "mode": "task1_safe_retract",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "base_motion_commanded": False,
        "published_control_topics": [],
    }
    node = None
    try:
        node = Ros2MissionNode(node_name="task1_safe_retract")
        node.wait_for_robot_state(timeout_sec=10.0)
        left, right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
        gripper_left = float(np.clip(raw_left_gripper, 0.0, 1.0))
        gripper_right = float(np.clip(raw_right_gripper, 0.0, 1.0))
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        located = load_position_reference(args.position_report, node, args.position_tolerance, args.yaw_tolerance)
        box_base = np.asarray(located["center_base"], dtype=float)
        contact_slide = float(PRE_GRASP_Z0 - (box_base[2] + TASK1_GRASP_Z_OFFSET_M))
        lower_slide = recovery_slide_target(current_slide, contact_slide)
        contact_center = box_base + np.array([args.current_grasp_fwd_offset, 0.0, TASK1_GRASP_Z_OFFSET_M])
        # First leave each box face laterally by 5 cm at the same forward and
        # vertical coordinates.  This is safer than moving rearward while the
        # box is still supported by the table.
        release_half = args.current_hold_half + args.lateral_release_margin
        release_left = contact_center + np.array([0.0, release_half, 0.0])
        release_right = contact_center + np.array([0.0, -release_half, 0.0])
        release_plan = solve_bimanual_pose(lower_slide, left, right, release_left, release_right)
        retract_plan = solve_bimanual_hug_pose(lower_slide, release_plan["left_joint_target"], release_plan["right_joint_target"])
        result.update({
            "position_reference": located,
            "initial_slide": current_slide,
            "contact_slide": contact_slide,
            "lower_slide": lower_slide,
            "home_slide": args.home_slide,
            "initial_raw_gripper_feedback": {"left": raw_left_gripper, "right": raw_right_gripper},
            "current_grasp_fwd_offset_m": args.current_grasp_fwd_offset,
            "current_hold_half_m": args.current_hold_half,
            "lateral_release_half_m": release_half,
            "lateral_release_plan": release_plan,
            "retract_plan": retract_plan,
        })
        if not args.apply:
            result["status"] = "dry_run"
            return 0

        # Keep the current arm pose while returning a possibly lifted box to
        # the tabletop.  Only then move the arms outward from the box.
        _traverse_spine(node, current_slide, lower_slide, args.spine_max_step, args.settle_timeout, 0.010, True, result)
        released_left, released_right = _traverse_pair(
            node, left, right, release_plan["left_joint_target"], release_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, 0.015,
            gripper_left, gripper_right, True, result,
        )
        retracted_left, retracted_right = _traverse_pair(
            node, released_left, released_right, retract_plan["left_joint_target"], retract_plan["right_joint_target"],
            args.joint_max_step, args.settle_timeout, 0.015,
            gripper_left, gripper_right, True, result,
        )
        _traverse_spine(node, lower_slide, args.home_slide, args.spine_max_step, args.settle_timeout, 0.010, True, result)
        _traverse_pair(
            node, retracted_left, retracted_right, np.zeros(6), np.zeros(6),
            args.joint_max_step, args.settle_timeout, 0.015,
            gripper_left, gripper_right, True, result,
        )
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
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
