"""Minimal real-server gripper feedback check: one gripper, then return.

The script holds all six arm joints at their measured positions, changes only
one gripper by a bounded positive delta, waits for JointState feedback, then
returns to its original value.  It is deliberately separate from contact and
transport tests.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from ros_contract import LEFT_ARM_COMMAND_TOPIC, RIGHT_ARM_COMMAND_TOPIC
from ros_sensor_utils import SensorCache


GRIPPER_LIMITS = (0.0, 1.0)
MAX_GRIPPER_DELTA = 0.15


def target_gripper_value(current: float, delta: float) -> float:
    """Build a bounded gripper-only target from valid joint feedback."""
    current = float(current)
    delta = float(delta)
    if not math.isfinite(current) or not GRIPPER_LIMITS[0] <= current <= GRIPPER_LIMITS[1]:
        raise ValueError("current gripper feedback must be within [0.0, 1.0]")
    if not math.isfinite(delta) or not 0.02 <= delta <= MAX_GRIPPER_DELTA:
        raise ValueError(f"gripper delta must be within [0.02, {MAX_GRIPPER_DELTA:.2f}]")
    target = current + delta
    if target > GRIPPER_LIMITS[1]:
        raise ValueError("requested gripper target exceeds the [0.0, 1.0] limit")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="One bounded gripper feedback check, with automatic return")
    parser.add_argument("--arm", choices=("l", "r"), required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--settle-timeout", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.010)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", default="/tmp/controlled_gripper_check.json")
    parser.add_argument("--apply", action="store_true", help="required before publishing a gripper command")
    args = parser.parse_args(argv)
    if args.settle_timeout <= 0 or args.tolerance <= 0 or args.hold_seconds < 0:
        parser.error("timeouts/tolerance must be positive and hold-seconds non-negative")

    prefix = "left" if args.arm == "l" else "right"
    joint_names = [f"{prefix}_arm_joint{i}" for i in range(1, 7)]
    gripper_name = f"{prefix}_arm_eef_gripper_joint"
    topic = LEFT_ARM_COMMAND_TOPIC if args.arm == "l" else RIGHT_ARM_COMMAND_TOPIC
    result = {
        "mode": "controlled_single_gripper_feedback_check",
        "arm": args.arm,
        "joint_name": gripper_name,
        "delta": args.delta,
        "apply": bool(args.apply),
        "published_control_topics": [],
        "other_arm_commanded": False,
        "base_head_spine_commanded": False,
        "six_arm_joints_preserved": True,
        "formal_motion_stays_disabled": True,
    }
    node = publisher = None
    initial_joints = None
    initial_gripper = target = None
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
        node = Node("controlled_gripper_check")
        node.create_subscription(JointState, "/joint_states", update, qos_profile_sensor_data)
        publisher = node.create_publisher(Float64MultiArray, topic, 10)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                initial_joints = cache.joint_vector(joint_names)
                initial_gripper = float(cache.joint_vector([gripper_name])[0])
                break
            except Exception:
                pass
        if initial_joints is None or initial_gripper is None:
            raise TimeoutError(f"timed out waiting for {prefix} arm/gripper JointState")
        target = target_gripper_value(initial_gripper, args.delta)
        result.update({
            "initial_arm_joints": initial_joints.tolist(),
            "initial_gripper": initial_gripper,
            "target_gripper": target,
        })
        if not args.apply:
            result["status"] = "dry_run"
            return 0

        def publish(gripper):
            message = Float64MultiArray()
            message.data = [float(value) for value in initial_joints] + [float(gripper)]
            publisher.publish(message)

        def wait_for(gripper):
            deadline = time.monotonic() + args.settle_timeout
            last_joints = last_gripper = None
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    last_joints = cache.joint_vector(joint_names)
                    last_gripper = float(cache.joint_vector([gripper_name])[0])
                except Exception:
                    continue
                joint_error = float(np.max(np.abs(last_joints - initial_joints)))
                if joint_error > args.tolerance:
                    raise RuntimeError(f"a preserved arm joint deviated by {joint_error:.4f} rad")
                if abs(last_gripper - gripper) <= args.tolerance:
                    return last_joints, last_gripper
            raise TimeoutError(f"{prefix} gripper did not reach target within {args.settle_timeout:.1f}s; last={last_gripper}")

        publish(target)
        command_issued = True
        result["published_control_topics"].append(topic)
        reached_joints, reached_gripper = wait_for(target)
        result["reached_gripper"] = reached_gripper
        result["reached_arm_joints"] = reached_joints.tolist()
        until = time.monotonic() + args.hold_seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=min(0.05, until - time.monotonic()))
        publish(initial_gripper)
        returned_joints, returned_gripper = wait_for(initial_gripper)
        result["returned_gripper"] = returned_gripper
        result["returned_arm_joints"] = returned_joints.tolist()
        result["status"] = "passed"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if publisher is not None and command_issued and initial_joints is not None and initial_gripper is not None:
            try:
                message = Float64MultiArray()
                message.data = [float(value) for value in initial_joints] + [float(initial_gripper)]
                publisher.publish(message)
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
