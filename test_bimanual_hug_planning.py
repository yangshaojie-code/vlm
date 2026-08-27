import unittest

import numpy as np

from bimanual_hug_planning import (
    FK_POSITION_TOLERANCE_M,
    LEGACY_FAR_LIP_GRASP_FWD_M,
    MAX_INWARD_DELTA,
    PRE_GRASP_LAT,
    build_bimanual_hug_plan,
    lateral_for_inward,
    side_face_hug_geometry,
    solve_bimanual_pose,
)


class BimanualHugPlanningTests(unittest.TestCase):
    def test_maximum_inward_plan_is_symmetric_and_fk_verified(self):
        plan = build_bimanual_hug_plan(
            slide=0.02,
            left_current=np.zeros(6),
            right_current=np.zeros(6),
            inward_delta=MAX_INWARD_DELTA,
            max_step=0.08,
        )
        pregrasp = plan["pregrasp"]
        inward = plan["inward"]
        self.assertIsNotNone(inward)
        self.assertEqual(plan["inward_waypoint_count"], 5)
        self.assertAlmostEqual(inward["lateral_offset_m"], PRE_GRASP_LAT - MAX_INWARD_DELTA)
        self.assertAlmostEqual(inward["left_target_position"][0], inward["right_target_position"][0])
        self.assertAlmostEqual(inward["left_target_position"][1], -inward["right_target_position"][1])
        self.assertAlmostEqual(inward["left_target_position"][2], pregrasp["left_target_position"][2])
        self.assertLess(inward["left_fk_error_m"], FK_POSITION_TOLERANCE_M)
        self.assertLess(inward["right_fk_error_m"], FK_POSITION_TOLERANCE_M)

    def test_inward_limit_is_rejected_before_kinematic_execution(self):
        with self.assertRaises(ValueError):
            lateral_for_inward(MAX_INWARD_DELTA + 0.001)
        with self.assertRaises(ValueError):
            build_bimanual_hug_plan(0.02, np.zeros(6), np.zeros(6), inward_delta=0.05)

    def test_arbitrary_precontact_targets_are_fk_verified(self):
        plan = solve_bimanual_pose(
            0.442,
            np.zeros(6),
            np.zeros(6),
            [0.49, 0.16, 0.879],
            [0.49, -0.16, 0.879],
        )
        self.assertLess(plan["left_fk_error_m"], FK_POSITION_TOLERANCE_M)
        self.assertLess(plan["right_fk_error_m"], FK_POSITION_TOLERANCE_M)

    def test_side_face_hug_centers_palms_on_the_squeezed_faces(self):
        table = side_face_hug_geometry(squeeze_axis="x", box_point="center")
        self.assertAlmostEqual(table["grasp_fwd_offset_m"], 0.0)
        self.assertAlmostEqual(table["grasp_z_offset_m"], 0.02)
        self.assertAlmostEqual(table["hold_half_m"], 0.115)
        self.assertAlmostEqual(table["approach_half_m"], 0.13)
        front = side_face_hug_geometry(squeeze_axis="x", box_point="front_face")
        self.assertAlmostEqual(front["grasp_fwd_offset_m"], 0.08)
        self.assertGreater(LEGACY_FAR_LIP_GRASP_FWD_M, 0.5 * table["depth_half_m"])
        shelf = side_face_hug_geometry(squeeze_axis="y", box_point="center")
        self.assertAlmostEqual(shelf["grasp_fwd_offset_m"], 0.0)
        self.assertAlmostEqual(shelf["hold_half_m"], 0.08)
        self.assertAlmostEqual(shelf["approach_half_m"], 0.10)
        with self.assertRaises(ValueError):
            side_face_hug_geometry(squeeze_axis="z")


if __name__ == "__main__":
    unittest.main()
