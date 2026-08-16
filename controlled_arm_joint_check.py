"""Minimal real-server arm-channel check: one joint, one arm, then return.

This is deliberately narrower than an IK pose test.  It publishes exactly one
arm command topic, keeps the other five joints and the gripper at feedback
values, and requires ``--apply`` before the first command.  No base, head,
spine, opposite arm, or gripper change is commanded.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from ros_contract import LEFT_ARM_COMMAND_TOPIC, RIGHT_ARM_COMMAND_TOPIC
from ros_sensor_utils import SensorCache


ARM_LIMITS = np.array([
    [-3.151, 2.080], [-2.963, 0.181], [-0.094, 3.161],
    [-3.012, 3.012], [-1.859, 1.859], [-3.017, 3.017],
], dtype=float)


def target_joint_vector(current, joint_index: int, delta: float) -> np.ndarray:
    values = np.asarray(current, dtype=float)
    joint_index = int(joint_index)
    delta = float(delta)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("current arm feedback must contain six finite joints")
    if not 1 <= joint_index <= 6:
        raise ValueError("joint-index must be in [1, 6]")
    if not math.isfinite(delta) or abs(delta) > 0.08:
        raise ValueError("one joint test is limited to +/-0.08 rad")
    result = values.copy()
    result[joint_index - 1] += delta
    if np.any(result < ARM_LIMITS[:, 0]) or np.any(result > ARM_LIMITS[:, 1]):
        raise ValueError("target arm vector exceeds official joint limits")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="One bounded arm-joint feedback check, with automatic return")
    parser.add_argument("--arm", choices=("l", "r"), required=True)
    parser.add_argument("--joint-index", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--settle-timeout", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.010)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", default="/tmp/controlled_arm_joint_check.json")
    parser.add_argument("--apply", action="store_true", help="required before publishing a joint command")
    args = parser.parse_args(argv)
    if args.settle_timeout <= 0 or args.tolerance <= 0 or args.hold_seconds < 0:
        parser.error("timeouts/tolerance must be positive and hold-seconds non-negative")

    prefix = "left" if args.arm == "l" else "right"
    joint_names = [f"{prefix}_arm_joint{i}" for i in range(1, 7)]
    gripper_name = f"{prefix}_arm_eef_gripper_joint"
    result = {
        "mode": "controlled_single_arm_joint_feedback_check",
        "arm": args.arm,
        "joint_name": joint_names[args.joint_index - 1] if 1 <= args.joint_index <= 6 else None,
        "joint_index": args.joint_index,
        "delta": args.delta,
        "apply": bool(args.apply),
        "published_control_topics": [],
        "other_arm_commanded": False,
        "base_head_spine_commanded": False,
        "formal_motion_stays_disabled": True,
    }
    node = publisher = None
    initial = target = None
    gripper = None
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
        node = Node("controlled_arm_joint_check")
        node.create_subscription(JointState, "/joint_states", update, qos_profile_sensor_data)
        topic = LEFT_ARM_COMMAND_TOPIC if args.arm == "l" else RIGHT_ARM_COMMAND_TOPIC
        publisher = node.create_publisher(Float64MultiArray, topic, 10)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                initial = cache.joint_vector(joint_names)
                gripper = float(cache.joint_vector([gripper_name])[0])
                break
            except Exception:
                pass
        if initial is None or gripper is None:
            raise TimeoutError(f"timed out waiting for {prefix} arm JointState")
        if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError(f"invalid current gripper feedback: {gripper!r}")
        target = target_joint_vector(initial, args.joint_index, args.delta)
        result.update({"initial": initial.tolist(), "target": target.tolist(), "preserved_gripper": gripper})
        if not args.apply:
            result["status"] = "dry_run"
            return 0

        def publish(values):
            message = Float64MultiArray()
            message.data = [float(value) for value in values] + [gripper]
            publisher.publish(message)

        def wait_for(values):
            deadline = time.monotonic() + args.settle_timeout
            last = None
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    last = cache.joint_vector(joint_names)
                except Exception:
                    continue
                if np.max(np.abs(last - values)) <= args.tolerance:
                    return last
            raise TimeoutError(
                f"{prefix} arm did not reach bounded target within {args.settle_timeout:.1f}s; last={last}"
            )

        publish(target)
        command_issued = True
        result["published_control_topics"].append(topic)
        result["reached_target"] = wait_for(target).tolist()
        until = time.monotonic() + args.hold_seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=min(0.05, until - time.monotonic()))
        publish(initial)
        result["returned"] = wait_for(initial).tolist()
        result["status"] = "passed"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if publisher is not None and command_issued and initial is not None and gripper is not None:
            try:
                message = Float64MultiArray()
                message.data = [float(value) for value in initial] + [float(gripper)]
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
