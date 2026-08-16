import unittest

import numpy as np

from bimanual_pregrasp_check import (
    LEFT_A_ROT,
    PRE_GRASP_FWD,
    PRE_GRASP_LAT,
    PRE_GRASP_Z0,
    RIGHT_A_ROT,
    pregrasp_targets,
    solve_plan,
    waypoint_count,
)


class BimanualPregraspCheckTests(unittest.TestCase):
    def test_target_geometry_tracks_slide(self):
        left, right = pregrasp_targets(0.10)
        np.testing.assert_allclose(left, [PRE_GRASP_FWD, PRE_GRASP_LAT, PRE_GRASP_Z0 - 0.10])
        np.testing.assert_allclose(right, [PRE_GRASP_FWD, -PRE_GRASP_LAT, PRE_GRASP_Z0 - 0.10])
        self.assertAlmostEqual(np.linalg.det(LEFT_A_ROT), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.det(RIGHT_A_ROT), 1.0, places=5)

    def test_official_kdl_plan_round_trips_with_fk(self):
        plan = solve_plan(0.02, np.zeros(6), np.zeros(6))
        self.assertLess(plan["left_fk_error_m"], 1e-6)
        self.assertLess(plan["right_fk_error_m"], 1e-6)

    def test_waypoints_bound_joint_delta(self):
        self.assertEqual(waypoint_count(np.zeros(6), np.full(6, 0.21), np.zeros(6), np.zeros(6), 0.10), 3)

    def test_reference_posture_is_no_contact_height(self):
        left, right = pregrasp_targets(0.02)
        self.assertGreater(left[2], 1.30)
        self.assertAlmostEqual(left[0], right[0])
        self.assertAlmostEqual(left[1], -right[1])

    def test_limited_inward_posture_remains_high_and_symmetric(self):
        left, right = pregrasp_targets(0.02, PRE_GRASP_LAT - 0.04)
        self.assertGreater(left[2], 1.30)
        self.assertAlmostEqual(left[1], 0.18475936)
        self.assertAlmostEqual(right[1], -0.18475936)

    def test_rejects_excessive_inward_geometry(self):
        with self.assertRaises(ValueError):
            pregrasp_targets(0.02, PRE_GRASP_LAT - 0.05)


if __name__ == "__main__":
    unittest.main()
