import unittest

from mission_protocol import TaskSpec, parse_mission_payload
from task_targets import remember_task1_source, resolve_place_world, validate_task_context
from test_mission_protocol import PAYLOAD


class TaskTargetTests(unittest.TestCase):
    def test_task_two_uses_persisted_task_one_source(self):
        mission = parse_mission_payload(PAYLOAD)
        context = {}
        remember_task1_source(context, [1, 2, 0.3], "right")
        task = mission.task(2)
        self.assertEqual(resolve_place_world(task, context), (1.0, 2.0, 0.8))
        validate_task_context(task, context)

    def test_missing_placement_radius_fails_closed(self):
        task = TaskSpec(task=1, instruction="x", target_color="pink", place_world=(1, 2, 3))
        with self.assertRaises(ValueError):
            validate_task_context(task, {})


if __name__ == "__main__":
    unittest.main()
