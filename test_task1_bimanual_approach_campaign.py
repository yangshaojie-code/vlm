import unittest

from task1_bimanual_approach_campaign import (
    bounded_gripper_feedback,
    clearance_schedule,
    command_gripper_value,
    gripper_waypoints,
)


class Task1BimanualApproachCampaignTests(unittest.TestCase):
    def test_clearance_schedule_has_positive_bounded_inward_steps(self):
        self.assertEqual(clearance_schedule(0.03, 0.02, 0.01), [0.03, 0.02])
        with self.assertRaises(ValueError):
            clearance_schedule(0.03, 0.01, 0.01)

    def test_gripper_waypoints_open_in_bounded_synchronized_steps(self):
        points = gripper_waypoints(0.005, 0.007, 1.0, 0.10)
        self.assertEqual(len(points), 10)
        self.assertEqual(points[-1], (1.0, 1.0))
        prior = (0.005, 0.007)
        for point in points:
            self.assertLessEqual(abs(point[0] - prior[0]), 0.10 + 1e-12)
            self.assertLessEqual(abs(point[1] - prior[1]), 0.10 + 1e-12)
            prior = point

    def test_invalid_gripper_step_is_rejected(self):
        with self.assertRaises(ValueError):
            gripper_waypoints(0.0, 0.0, 1.0, 0.16)

    def test_endpoint_feedback_is_clamped_for_all_future_commands(self):
        self.assertEqual(bounded_gripper_feedback(1.005), 1.0)
        self.assertEqual(command_gripper_value(1.005), 1.0)
        self.assertEqual(bounded_gripper_feedback(1.03), 1.0)


if __name__ == "__main__":
    unittest.main()
