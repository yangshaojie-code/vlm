"""Official Client-container entry point for one continuous three-task game."""

import argparse

from ros2_mission_node import Ros2MissionNode
from ros_mission_executor import RosMissionExecutor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the formal three-task ROS 2 Client")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate feedback, RGB-D and TF without publishing motion commands",
    )
    args = parser.parse_args(argv)
    node = Ros2MissionNode()
    stop_robot_on_close = not args.preflight_only
    try:
        mission = node.wait_for_mission(timeout_sec=20.0)
        node.node.get_logger().info(f"loaded {len(mission.tasks)} formal tasks")
        executor = RosMissionExecutor(node)
        executor.preflight(timeout_sec=10.0)
        if args.preflight_only:
            node.node.get_logger().info("formal preflight passed; no motion command was sent")
            return 0
        if not executor.motion_ready:
            raise RuntimeError(
                f"formal motion is blocked: {executor.motion_block_reason}; run --preflight-only"
            )
        # The instruction callback already starts the state machine. Drive only
        # attempts here so a late callback cannot reset the physical scene.
        while node.orchestrator.state.value not in ("GAME_DONE", "TIMEOUT"):
            task = node.orchestrator.start_attempt()
            result = executor.execute_attempt(task, node.orchestrator.context, node.orchestrator.current_attempt)
            node.orchestrator.complete_attempt(result.success, result.score, result.reason)
            node.orchestrator.settle()
        return 0
    finally:
        node.close(stop_robot=stop_robot_on_close)


if __name__ == "__main__":
    raise SystemExit(main())
