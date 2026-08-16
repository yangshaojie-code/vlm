import unittest

from controlled_motion_check import bounded_target


class ControlledMotionCheckTests(unittest.TestCase):
    def test_builds_small_in_range_target(self):
        self.assertAlmostEqual(bounded_target("head_yaw", 0.0, 0.10), 0.10)
        self.assertAlmostEqual(bounded_target("spine", 0.10, -0.05), 0.05)

    def test_rejects_large_or_out_of_range_motion(self):
        with self.assertRaises(ValueError):
            bounded_target("head_pitch", 0.0, 0.16)
        with self.assertRaises(ValueError):
            bounded_target("spine", 0.85, 0.05)


if __name__ == "__main__":
    unittest.main()
