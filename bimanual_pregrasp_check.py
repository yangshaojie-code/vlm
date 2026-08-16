"""Official-KDL, no-contact bimanual pre-grasp posture check.

The posture is the fixed Task 1 reference pre-grasp position: both end
effectors are high and only 0.48 m in front of ``base_link``.  With the robot
at its start pose it is well short of the table, so this test validates paired
arm control and KDL/FK agreement without approaching a box.  The tool never
commands base, head, spine, or gripper motion and requires ``--apply``.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import (
    LEFT_A_ROT,
    MAX_INWARD_DELTA,
    PRE_GRASP_FWD,
    PRE_GRASP_LAT,
    PRE_GRASP_Z0,
    RIGHT_A_ROT,
    build_bimanual_hug_plan,
    pregrasp_targets,
    solve_bimanual_hug_pose,
    waypoint_count,
)
from ros_contract import LEFT_ARM_COMMAND_TOPIC, RIGHT_ARM_COMMAND_TOPIC
from ros_sensor_utils import SensorCache


# Compatibility name used by the existing CLI tests and callers.
solve_plan = solve_bimanual_hug_pose


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paired no-contact KDL pre-grasp check with automatic return")
    parser.add_argument("--max-step", type=float, default=0.10, help="largest per-waypoint arm-joint delta in rad")
    parser.add_argument("--settle-timeout", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.010)
    parser.add_argument(
        "--stable-samples", type=int, default=3,
        help="consecutive in-tolerance JointState samples required at every waypoint",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument(
        "--inward-delta", type=float, default=0.0,
        help="optional high no-contact inward move per arm after pre-grasp, limited to 0.04 m",
    )
    parser.add_argument("--output", default="/tmp/bimanual_pregrasp_check.json")
    parser.add_argument("--apply", action="store_true", help="required before publishing paired arm commands")
    args = parser.parse_args(argv)
    if not 0.02 <= args.max_step <= 0.15:
        parser.error("max-step must be within [0.02, 0.15]")
    if args.settle_timeout <= 0 or args.tolerance <= 0 or args.hold_seconds < 0 or args.stable_samples <= 0:
        parser.error("timeouts/tolerance must be positive and hold-seconds non-negative")
    if not 0.0 <= args.inward_delta <= MAX_INWARD_DELTA:
        parser.error("inward-delta must be within [0.0, 0.04] m")

    result = {
        "mode": "official_kdl_bimanual_no_contact_pregrasp_check",
        "apply": bool(args.apply),
        "published_control_topics": [],
        "base_head_spine_commanded": False,
        "gripper_motion_commanded": False,
        "formal_motion_stays_disabled": True,
        "tolerance": args.tolerance,
        "stable_samples_required": args.stable_samples,
        "inward_delta_m": args.inward_delta,
    }
    node = left_pub = right_pub = None
    initial_left = initial_right = None
    left_gripper = right_gripper = None
    command_issued = False
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        cache = SensorCache()

        def update(message):
            try:
                cache.update_joint_state(message)
            except Exception as exc:
                result.setdefault("feedback_errors", []).append(str(exc))

        rclpy.init()
        node = Node("bimanual_pregrasp_check")
        node.create_subscription(JointState, "/joint_states", update, qos_profile_sensor_data)
        left_pub = node.create_publisher(Float64MultiArray, LEFT_ARM_COMMAND_TOPIC, 10)
        right_pub = node.create_publisher(Float64MultiArray, RIGHT_ARM_COMMAND_TOPIC, 10)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                initial_left = cache.joint_vector([f"left_arm_joint{i}" for i in range(1, 7)])
                initial_right = cache.joint_vector([f"right_arm_joint{i}" for i in range(1, 7)])
                slide = float(cache.joint_vector(["slide_joint"])[0])
                left_gripper = float(cache.joint_vector(["left_arm_eef_gripper_joint"])[0])
                right_gripper = float(cache.joint_vector(["right_arm_eef_gripper_joint"])[0])
                break
            except Exception:
                pass
        if any(value is None for value in (initial_left, initial_right, left_gripper, right_gripper)):
            raise TimeoutError("timed out waiting for paired arm/slide/gripper JointState")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (left_gripper, right_gripper)):
            raise ValueError("current gripper feedback must be finite and within [0, 1]")
        hug_plan = build_bimanual_hug_plan(
            slide,
            initial_left,
            initial_right,
            inward_delta=args.inward_delta,
            max_step=args.max_step,
        )
        plan = hug_plan["pregrasp"]
        left_target = np.asarray(plan["left_joint_target"])
        right_target = np.asarray(plan["right_joint_target"])
        inward_plan = hug_plan["inward"]
        stages = hug_plan["pregrasp_waypoint_count"]
        inward_stages = hug_plan["inward_waypoint_count"]
        result.update({
            "initial_left": initial_left.tolist(), "initial_right": initial_right.tolist(),
            "preserved_left_gripper": left_gripper, "preserved_right_gripper": right_gripper,
            "waypoint_count": stages, "inward_waypoint_count": inward_stages, **plan,
        })
        if inward_plan is not None:
            result["inward_plan"] = inward_plan
        if not args.apply:
            result["status"] = "dry_run"
            return 0

        def publish_pair(left, right):
            left_message = Float64MultiArray()
            right_message = Float64MultiArray()
            left_message.data = [float(value) for value in left] + [left_gripper]
            right_message.data = [float(value) for value in right] + [right_gripper]
            left_pub.publish(left_message)
            right_pub.publish(right_message)

        def wait_pair(left, right):
            deadline = time.monotonic() + args.settle_timeout
            last_left = last_right = None
            stable = 0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    last_left = cache.joint_vector([f"left_arm_joint{i}" for i in range(1, 7)])
                    last_right = cache.joint_vector([f"right_arm_joint{i}" for i in range(1, 7)])
                except Exception:
                    continue
                if max(np.max(np.abs(last_left - left)), np.max(np.abs(last_right - right))) <= args.tolerance:
                    stable += 1
                    if stable >= args.stable_samples:
                        return last_left, last_right
                else:
                    stable = 0
            raise TimeoutError(f"paired arm feedback did not reach waypoint; left={last_left}, right={last_right}")

        result["published_control_topics"] = [LEFT_ARM_COMMAND_TOPIC, RIGHT_ARM_COMMAND_TOPIC]
        def traverse(left_start, right_start, left_end, right_end, count):
            nonlocal command_issued
            reached_left = reached_right = None
            for stage in range(1, count + 1):
                fraction = stage / count
                left_waypoint = left_start + fraction * (left_end - left_start)
                right_waypoint = right_start + fraction * (right_end - right_start)
                publish_pair(left_waypoint, right_waypoint)
                command_issued = True
                reached_left, reached_right = wait_pair(left_waypoint, right_waypoint)
            return reached_left, reached_right

        left_reached, right_reached = traverse(initial_left, initial_right, left_target, right_target, stages)
        result["reached_pregrasp_left"] = left_reached.tolist()
        result["reached_pregrasp_right"] = right_reached.tolist()
        if inward_plan is not None:
            inner_left = np.asarray(inward_plan["left_joint_target"])
            inner_right = np.asarray(inward_plan["right_joint_target"])
            left_inner, right_inner = traverse(left_target, right_target, inner_left, inner_right, inward_stages)
            result["reached_inward_left"] = left_inner.tolist()
            result["reached_inward_right"] = right_inner.tolist()
        hold_until = time.monotonic() + args.hold_seconds
        while time.monotonic() < hold_until:
            rclpy.spin_once(node, timeout_sec=min(0.05, hold_until - time.monotonic()))
        if inward_plan is not None:
            traverse(inner_left, inner_right, left_target, right_target, inward_stages)
        returned_left, returned_right = traverse(left_target, right_target, initial_left, initial_right, stages)
        result["returned_left"] = returned_left.tolist()
        result["returned_right"] = returned_right.tolist()
        result["status"] = "passed"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if command_issued and left_pub is not None and initial_left is not None and initial_right is not None:
            try:
                left_message = Float64MultiArray()
                right_message = Float64MultiArray()
                left_message.data = [float(value) for value in initial_left] + [left_gripper]
                right_message.data = [float(value) for value in initial_right] + [right_gripper]
                left_pub.publish(left_message)
                right_pub.publish(right_message)
                result["return_after_failure_published"] = True
            except Exception as restore_exc:
                result["return_after_failure_error"] = str(restore_exc)
        return 2
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"could not write report {args.output}: {exc}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if node is not None:
            node.destroy_node()
            try:
                import rclpy
                if rclpy.ok():
                    rclpy.shutdown()
            except ImportError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
