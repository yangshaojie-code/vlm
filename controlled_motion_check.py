"""Explicit, bounded real-server joint feedback check.

This tool is intentionally limited to head yaw, head pitch, and spine slide.
It never commands the base, arms, or grippers, and it requires ``--apply``
before publishing one controlled motion. Formal mission motion remains
disabled independently in ``RosMissionExecutor``.
"""

import argparse
import json
import math
from pathlib import Path
import time

from head_camera_kinematics import HEAD_PITCH_LIMITS, HEAD_YAW_LIMITS, SLIDE_LIMITS
from ros2_mission_node import Ros2MissionNode


CHANNELS = {
    "head_yaw": ("head_yaw_joint", HEAD_YAW_LIMITS, 0.10),
    "head_pitch": ("head_pitch_joint", HEAD_PITCH_LIMITS, -0.08),
    "spine": ("slide_joint", SLIDE_LIMITS, 0.05),
}


def bounded_target(channel: str, current: float, delta: float) -> float:
    """Build a small target and reject, rather than clamp, unsafe requests."""
    if channel not in CHANNELS:
        raise ValueError(f"unsupported channel: {channel}")
    _name, limits, _default = CHANNELS[channel]
    current, delta = float(current), float(delta)
    if not math.isfinite(current) or not math.isfinite(delta):
        raise ValueError("current joint value and delta must be finite")
    if abs(delta) > 0.15:
        raise ValueError("one controlled motion is limited to +/-0.15 rad/m")
    target = current + delta
    if not limits[0] <= target <= limits[1]:
        raise ValueError(f"target {target:.4f} is outside {channel} limits {limits}")
    return target


def wait_for_joint(node: Ros2MissionNode, joint_name: str, target: float, timeout: float, tolerance: float) -> float:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        try:
            last = float(node.sensors.joint_vector([joint_name])[0])
        except Exception:
            continue
        if abs(last - target) <= tolerance:
            return last
    raise TimeoutError(
        f"{joint_name} feedback did not reach target={target:.4f} within {timeout:.1f}s; last={last!r}"
    )


def command_channel(node: Ros2MissionNode, channel: str, target: float) -> None:
    if channel == "spine":
        node.controller.command_spine(target)
        return
    values = node.sensors.joint_vector(["head_yaw_joint", "head_pitch_joint"]).tolist()
    index = 0 if channel == "head_yaw" else 1
    values[index] = target
    node.controller.command_head(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bounded real-server feedback check; no base or arm motion")
    parser.add_argument("--channel", choices=sorted(CHANNELS), default="head_yaw")
    parser.add_argument("--delta", type=float, help="relative joint target; maximum magnitude is 0.15")
    parser.add_argument("--settle-timeout", type=float, default=6.0)
    parser.add_argument(
        "--tolerance", type=float, default=0.010,
        help="maximum feedback error at target and return (default: 0.010)",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", default="/tmp/controlled_motion_check.json")
    parser.add_argument("--apply", action="store_true", help="required before publishing the motion")
    args = parser.parse_args(argv)
    if args.settle_timeout <= 0 or args.tolerance <= 0 or args.hold_seconds < 0:
        parser.error("timeouts/tolerance must be positive and hold-seconds non-negative")

    joint_name, _limits, default_delta = CHANNELS[args.channel]
    delta = default_delta if args.delta is None else args.delta
    result = {
        "mode": "controlled_real_joint_feedback_check",
        "channel": args.channel,
        "joint_name": joint_name,
        "delta": delta,
        "apply": bool(args.apply),
        "published_control_channels": [],
        "formal_motion_stays_disabled": True,
    }
    node = None
    initial = None
    command_issued = False
    try:
        node = Ros2MissionNode(node_name="controlled_motion_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial = float(node.sensors.joint_vector([joint_name])[0])
        target = bounded_target(args.channel, initial, delta)
        result.update({"initial": initial, "target": target})
        if not args.apply:
            result["status"] = "dry_run"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        command_channel(node, args.channel, target)
        command_issued = True
        result["published_control_channels"].append(args.channel)
        result["reached_target"] = wait_for_joint(
            node, joint_name, target, args.settle_timeout, args.tolerance
        )
        hold_deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < hold_deadline:
            node.spin_once(min(0.05, hold_deadline - time.monotonic()))
        command_channel(node, args.channel, initial)
        result["returned"] = wait_for_joint(
            node, joint_name, initial, args.settle_timeout, args.tolerance
        )
        result["status"] = "passed"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if node is not None and command_issued and initial is not None:
            try:
                command_channel(node, args.channel, initial)
                result["return_after_failure_published"] = True
            except Exception as restore_exc:
                result["return_after_failure_error"] = str(restore_exc)
        return 2
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output = Path(args.output)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"could not write report {output}: {exc}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if node is not None:
            # This tool never commands base velocity, arms, or grippers.  Do
            # not call stop_all here because it publishes joint hold commands
            # on every channel, which would widen this test's authority.
            node.close(stop_robot=False)


if __name__ == "__main__":
    raise SystemExit(main())
