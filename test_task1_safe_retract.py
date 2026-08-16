import unittest

from task1_safe_retract import recovery_slide_target


class Task1SafeRetractTests(unittest.TestCase):
    def test_lowers_lifted_box_before_arm_retraction(self):
        self.assertAlmostEqual(recovery_slide_target(0.344, 0.444), 0.444)

    def test_does_not_raise_box_when_already_lower_than_contact_target(self):
        self.assertAlmostEqual(recovery_slide_target(0.500, 0.444), 0.500)


if __name__ == "__main__":
    unittest.main()
