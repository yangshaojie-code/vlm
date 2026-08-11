import json
import threading
import unittest
from types import SimpleNamespace

from mission_orchestrator import MissionOrchestrator, MissionState
from ros2_mission_node import Ros2MissionNode


def mission_payload(color="pink"):
    return json.dumps([
        {"task": 1, "instruction": "one", "target_color": color},
        {"task": 2, "instruction": "two", "target_color": "yellow"},
        {"task": 3, "instruction": "three", "target_color": "brown"},
    ])


class Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, value):
        self.warnings.append(str(value))

    def error(self, value):
        self.errors.append(str(value))


class MissionNodeCallbackTests(unittest.TestCase):
    def make_node(self):
        result = Ros2MissionNode.__new__(Ros2MissionNode)
        result.orchestrator = MissionOrchestrator()
        result.latest_instruction_raw = None
        result.errors = []
        result._mission_condition = threading.Condition()
        result.node = SimpleNamespace(get_logger=lambda: Logger())
        return result

    def test_repeated_instruction_does_not_restart_finished_mission(self):
        node = self.make_node()
        message = SimpleNamespace(data=mission_payload())
        node._instruction_callback(message)
        node.orchestrator.context["task1_source_world"] = (1.0, 2.0, 3.0)
        node.orchestrator.state = MissionState.GAME_DONE

        node._instruction_callback(message)

        self.assertEqual(node.orchestrator.state, MissionState.GAME_DONE)
        self.assertEqual(node.orchestrator.context["task1_source_world"], (1.0, 2.0, 3.0))

    def test_different_instruction_is_ignored_during_active_attempt(self):
        node = self.make_node()
        node._instruction_callback(SimpleNamespace(data=mission_payload()))
        node._instruction_callback(SimpleNamespace(data=mission_payload("yellow")))
        self.assertEqual(node.orchestrator.mission.task(1).target_color, "pink")

    def test_robot_state_wait_processes_callbacks_before_returning(self):
        node = self.make_node()
        required = ["slide_joint", "head_yaw_joint", "head_pitch_joint"]
        required += [f"left_arm_joint{i}" for i in range(1, 7)]
        required += ["left_arm_eef_gripper_joint"]
        required += [f"right_arm_joint{i}" for i in range(1, 7)]
        required += ["right_arm_eef_gripper_joint"]
        node.sensors = SimpleNamespace(odom=None, joint_names=())

        def spin_once(_timeout):
            node.sensors.odom = object()
            node.sensors.joint_names = tuple(required)

        node.spin_once = spin_once
        node.wait_for_robot_state(timeout_sec=0.1)

    def test_robot_state_wait_has_finite_timeout(self):
        node = self.make_node()
        node.sensors = SimpleNamespace(odom=None, joint_names=())
        node.spin_once = lambda _timeout: None
        with self.assertRaisesRegex(TimeoutError, "robot state timeout"):
            node.wait_for_robot_state(timeout_sec=0.0)


if __name__ == "__main__":
    unittest.main()
