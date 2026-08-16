import unittest

from controlled_gripper_check import target_gripper_value


class ControlledGripperCheckTests(unittest.TestCase):
    def test_builds_bounded_positive_target(self):
        self.assertAlmostEqual(target_gripper_value(0.006, 0.10), 0.106)

    def test_rejects_invalid_feedback_delta_and_limit(self):
        with self.assertRaises(ValueError):
            target_gripper_value(-0.01, 0.10)
        with self.assertRaises(ValueError):
            target_gripper_value(0.006, 0.01)
        with self.assertRaises(ValueError):
            target_gripper_value(0.95, 0.10)


if __name__ == "__main__":
    unittest.main()
