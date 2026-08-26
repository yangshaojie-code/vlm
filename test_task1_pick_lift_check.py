import unittest

import numpy as np

from task1_pick_lift_check import (
    APPROACH_JOINT_TOLERANCE_RAD,
    CARRY_EMPTY_JOINT_RESIDUAL_RAD,
    CONTACT_MIN_JOINT_RESIDUAL_RAD,
    TABLE_BLOCKED_HUG_MAX_RAD,
    TABLE_CONTACT_CLEARANCE_MAX_M,
    TASK1_HOLD_HALF_M,
    blocked_table_hug_lock,
    classify_approach_sample,
    contact_approach_geometry,
    contact_clearance_schedule,
    contact_residuals,
    carry_hold_ok,
    carry_squeeze_ok,
    holding_pose_ok,
    hug_moved_from_pregrasp,
    lift_slide_target,
)


class Task1PickLiftCheckTests(unittest.TestCase):
    def test_non_contact_approach_tolerance_has_small_feedback_margin(self):
        self.assertGreater(APPROACH_JOINT_TOLERANCE_RAD, 0.010)
        self.assertLessEqual(APPROACH_JOINT_TOLERANCE_RAD, CONTACT_MIN_JOINT_RESIDUAL_RAD)

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

    def test_settled_carry_pose_is_still_a_valid_hold(self):
        hold_left = np.array([0.0, -0.93, 0.51, -1.53, -1.19, 1.60])
        hold_right = np.array([0.0, -0.80, 0.50, 1.53, 1.07, -1.59])
        settled_left = hold_left + 0.008
        settled_right = hold_right - 0.008
        result = holding_pose_ok(settled_left, settled_right, hold_left, hold_right)
        self.assertLess(result["left_max_joint_residual_rad"], CONTACT_MIN_JOINT_RESIDUAL_RAD)
        self.assertFalse(result["symmetric_contact"])
        self.assertTrue(result["holding"])

    def test_hold_is_lost_when_one_arm_leaves_the_carry_pose(self):
        hold = np.zeros(6)
        result = holding_pose_ok([0.09, 0, 0, 0, 0, 0], np.zeros(6), hold, hold)
        self.assertFalse(result["holding"])

    def test_empty_close_is_not_a_carry_squeeze(self):
        squeeze_left = np.array([0.0, -0.974, 0.54, -1.53, -1.20, 1.60])
        squeeze_right = np.array([0.0, -0.847, 0.52, 1.52, 1.09, -1.59])
        closed_left = squeeze_left + 0.008
        closed_right = squeeze_right - 0.008
        result = carry_squeeze_ok(closed_left, closed_right, squeeze_left, squeeze_right)
        self.assertLess(result["left_max_joint_residual_rad"], CONTACT_MIN_JOINT_RESIDUAL_RAD)
        self.assertFalse(result["holding"])

    def test_blocked_inward_pose_is_a_carry_squeeze(self):
        squeeze = np.zeros(6)
        blocked_left = np.array([0.04, 0, 0, 0, 0, 0])
        blocked_right = np.array([-0.035, 0, 0, 0, 0, 0])
        result = carry_squeeze_ok(blocked_left, blocked_right, squeeze, squeeze)
        self.assertTrue(result["symmetric_contact"])
        self.assertTrue(result["holding"])

    def test_offset_box_is_still_a_carry_hold(self):
        hold = np.zeros(6)
        result = carry_hold_ok(
            [0.04157, 0, 0, 0, 0, 0], [-0.01377, 0, 0, 0, 0, 0], hold, hold,
        )
        self.assertFalse(result["symmetric_contact"])
        self.assertGreaterEqual(result["right_max_joint_residual_rad"], CARRY_EMPTY_JOINT_RESIDUAL_RAD)
        self.assertTrue(result["holding"])
        empty = carry_hold_ok(
            [0.008, 0, 0, 0, 0, 0], [-0.006, 0, 0, 0, 0, 0], hold, hold,
        )
        self.assertFalse(empty["holding"])
        one_empty = carry_hold_ok(
            [0.042, 0, 0, 0, 0, 0], [-0.006, 0, 0, 0, 0, 0], hold, hold,
        )
        self.assertFalse(one_empty["holding"])

    def test_first_interpolation_lag_is_not_plan_contact(self):
        high_left = np.array([0.0, 0.0, 0.0, -1.510, -0.766, 1.570])
        high_right = np.array([0.0, 0.0, 0.0, 1.510, 0.766, -1.570])
        plan_left = np.array([0.0, -0.974, 0.539, -1.526, -1.200, 1.597])
        plan_right = np.array([0.0, -0.847, 0.525, 1.523, 1.088, -1.592])
        first_left = high_left + 0.05 * (plan_left - high_left)
        first_right = high_right + 0.05 * (plan_right - high_right)
        lagged_left = np.array([-0.00086, 0.00023, 0.00021, -1.522, -0.767, 1.569])
        lagged_right = np.array([0.00086, 0.00023, 0.00020, 1.522, 0.767, -1.569])

        sample = classify_approach_sample(
            lagged_left, lagged_right, first_left, first_right,
            plan_left, plan_right, APPROACH_JOINT_TOLERANCE_RAD,
        )
        self.assertFalse(sample["waypoint_reached"])
        self.assertTrue(
            CONTACT_MIN_JOINT_RESIDUAL_RAD
            <= sample["waypoint_left_max_joint_residual_rad"]
            <= 0.08
        )
        self.assertFalse(sample["plan_contact"])
        self.assertGreater(sample["left_max_joint_residual_rad"], 0.08)
        self.assertFalse(hug_moved_from_pregrasp(lagged_left, lagged_right, high_left, high_right))

    def test_near_clearance_pose_is_plan_contact_and_left_pregrasp(self):
        high_left = np.zeros(6)
        high_right = np.zeros(6)
        plan_left = np.array([0.0, -0.97, 0.0, 0.0, 0.0, 0.0])
        plan_right = np.array([0.0, -0.85, 0.0, 0.0, 0.0, 0.0])
        current_left = plan_left + np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
        current_right = plan_right + np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
        sample = classify_approach_sample(
            current_left, current_right, plan_left, plan_right,
            plan_left, plan_right, APPROACH_JOINT_TOLERANCE_RAD,
        )
        self.assertTrue(sample["plan_contact"])
        self.assertTrue(hug_moved_from_pregrasp(current_left, current_right, high_left, high_right))

    def test_offset_station_stall_at_one_cm_is_a_blocked_hug(self):
        self.assertGreater(TABLE_CONTACT_CLEARANCE_MAX_M, 0.01)
        self.assertLess(TABLE_CONTACT_CLEARANCE_MAX_M, 0.02)
        high_left = np.array([0.0, 0.0, 0.0, -1.510, -0.766, 1.570])
        high_right = np.array([0.0, 0.0, 0.0, 1.510, 0.766, -1.570])
        stalled_left = np.array([0.00140, -0.98844, 0.58383, -1.53108, -1.20272, 1.59764])
        stalled_right = np.array([-0.00116, -0.89748, 0.57194, 1.52999, 1.12574, -1.59379])
        plan_left = stalled_left + np.array([0.0, -0.11, 0.06, 0.0, 0.0, 0.0])
        plan_right = stalled_right + np.array([0.0, -0.09, 0.05, 0.0, 0.0, 0.0])
        locked = blocked_table_hug_lock(
            stalled_left, stalled_right, high_left, high_right, plan_left, plan_right,
        )
        self.assertIsNotNone(locked)
        self.assertTrue(locked["feedback"]["blocked_hug"])
        self.assertTrue(locked["feedback"]["contact_detected"])
        self.assertGreater(locked["feedback"]["left_max_joint_residual_rad"], APPROACH_JOINT_TOLERANCE_RAD)
        self.assertLessEqual(locked["feedback"]["left_max_joint_residual_rad"], TABLE_BLOCKED_HUG_MAX_RAD)
        np.testing.assert_allclose(locked["left"], stalled_left)
        self.assertIsNone(blocked_table_hug_lock(
            high_left, high_right, high_left, high_right, plan_left, plan_right,
        ))

    def test_one_sided_table_stall_is_not_a_blocked_hug(self):
        high_left = np.zeros(6)
        high_right = np.zeros(6)
        hugged_left = np.array([0.0, -0.90, 0.0, 0.0, 0.0, 0.0])
        hugged_right = np.array([0.0, -0.90, 0.0, 0.0, 0.0, 0.0])
        plan_left = np.array([0.0, -1.00, 0.0, 0.0, 0.0, 0.0])
        plan_right = np.array([0.0, -1.50, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(blocked_table_hug_lock(
            hugged_left, hugged_right, high_left, high_right, plan_left, plan_right,
        ))


if __name__ == "__main__":
    unittest.main()
