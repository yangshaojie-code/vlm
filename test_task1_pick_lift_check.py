import unittest

import numpy as np

from task1_pick_lift_check import (
    TASK1_HOLD_HALF_M,
    contact_approach_geometry,
    contact_clearance_schedule,
    contact_residuals,
    lift_slide_target,
)


class Task1PickLiftCheckTests(unittest.TestCase):
    def test_contact_schedule_reaches_contact_in_small_steps(self):
        self.assertEqual(contact_clearance_schedule(0.02, 0.01), [0.02, 0.01, 0.0])
        with self.assertRaises(ValueError):
            contact_clearance_schedule(0.01, 0.01)

    def test_lift_reduces_slide_by_the_requested_height(self):
        self.assertAlmostEqual(lift_slide_target(0.444, 0.10), 0.344)
        with self.assertRaises(ValueError):
            lift_slide_target(0.444, 0.13)

    def test_contact_geometry_ends_with_box_half_width_lateral_targets(self):
        left, right = contact_approach_geometry([0.58, 0.0, 0.834], 0.0, 0.065, 0.045, TASK1_HOLD_HALF_M)
        np.testing.assert_allclose(left, [0.645, 0.115, 0.879], atol=1e-12)
        np.testing.assert_allclose(right, [0.645, -0.115, 0.879], atol=1e-12)

    def test_contact_requires_bounded_residual_on_both_arms(self):
        target = np.zeros(6)
        result = contact_residuals(
            [0.03, 0, 0, 0, 0, 0], [-0.025, 0, 0, 0, 0, 0], target, target,
        )
        self.assertTrue(result["symmetric_contact"])
        self.assertFalse(result["target_reached"])
        result = contact_residuals([0.03, 0, 0, 0, 0, 0], np.zeros(6), target, target)
        self.assertFalse(result["symmetric_contact"])


if __name__ == "__main__":
    unittest.main()
