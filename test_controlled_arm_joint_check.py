import unittest

import numpy as np

from controlled_arm_joint_check import target_joint_vector


class ControlledArmJointCheckTests(unittest.TestCase):
    def test_changes_only_requested_joint(self):
        current = np.zeros(6)
        target = target_joint_vector(current, 4, 0.05)
        np.testing.assert_allclose(target, [0.0, 0.0, 0.0, 0.05, 0.0, 0.0])

    def test_rejects_large_or_out_of_limit_request(self):
        with self.assertRaises(ValueError):
            target_joint_vector(np.zeros(6), 1, 0.09)
        with self.assertRaises(ValueError):
            target_joint_vector(np.array([2.07, 0, 0, 0, 0, 0]), 1, 0.05)


if __name__ == "__main__":
    unittest.main()
