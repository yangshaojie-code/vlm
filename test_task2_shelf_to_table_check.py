import math
import unittest

import numpy as np

from task1_pick_lift_check import carry_hold_ok
from task2_shelf_to_table_check import (
    BOX_HALF_Y_M,
    BROWN_FIXED_WORLD,
    INSTRUCTION_PLACE_WORLD,
    L2_BOX_CENTER_Z,
    PICK_YAW,
    PLACE_YAW,
    SHELF_HOLD_HALF_M,
    SHELF_LIFT_HEIGHT_M,
    TABLE_PLACE_HELD_X_MAX,
    already_at_table_place,
    already_carrying_box,
    already_holding_on_shelf,
    already_on_pick_approach,
    blocked_hug_lock,
    box_inside_table_radius,
    box_on_table_top,
    center_from_shelf_front,
    inward_hold_from_blocked,
    lifted_box_clears_upper_board,
    local_carry_hold,
    maintain_carry_hold,
    pick_stand_from_box,
    place_stand_xy,
    shelf_hug_targets,
    snap_l2_box_center,
    table_place_held_center,
    table_placement_error,
    validate_brown_world,
    validate_table_place_world,
)


class Task2ShelfToTableCheckTests(unittest.TestCase):
    def test_fixed_brown_is_in_l2_window(self):
        np.testing.assert_allclose(validate_brown_world(BROWN_FIXED_WORLD), BROWN_FIXED_WORLD)
        with self.assertRaises(ValueError):
            validate_brown_world([-2.50, 0.778, 1.156])

    def test_table_place_world_is_the_original_pink_slot(self):
        np.testing.assert_allclose(validate_table_place_world(INSTRUCTION_PLACE_WORLD), INSTRUCTION_PLACE_WORLD)
        with self.assertRaises(ValueError):
            validate_table_place_world([-2.68, 0.778, 1.156])

    def test_shelf_front_offset_moves_west_by_half_depth(self):
        center = center_from_shelf_front([-2.51, 0.778, 0.837], PICK_YAW, 0.12)
        np.testing.assert_allclose(center, [-2.63, 0.778, 0.837], atol=1e-12)

    def test_pick_stand_is_clamped_before_the_cabinet(self):
        stand = pick_stand_from_box(BROWN_FIXED_WORLD, PICK_YAW)
        self.assertGreaterEqual(stand[0], -1.96)
        self.assertAlmostEqual(abs(stand[2]), math.pi)
        self.assertLess(abs(stand[1] - 0.778), 1e-12)

    def test_current_shelf_front_pose_skips_staging(self):
        stand = pick_stand_from_box(BROWN_FIXED_WORLD, PICK_YAW)
        stage = np.array([stand[0] + 0.50, stand[1], PICK_YAW])
        current = np.array([-1.685012688267801, 0.8047523711716857, -3.0724637324139623])
        self.assertTrue(already_on_pick_approach(current, stand, stage))

    def test_shelf_hug_uses_the_short_face_width(self):
        left, right = shelf_hug_targets([0.67, 0.0, 0.837], 0.0, 0.065, 0.045, SHELF_HOLD_HALF_M)
        np.testing.assert_allclose(left, [0.735, SHELF_HOLD_HALF_M, 0.882], atol=1e-12)
        np.testing.assert_allclose(right, [0.735, -SHELF_HOLD_HALF_M, 0.882], atol=1e-12)
        self.assertAlmostEqual(SHELF_HOLD_HALF_M, BOX_HALF_Y_M)

    def test_l2_lift_stays_below_pink_layer(self):
        self.assertTrue(lifted_box_clears_upper_board(0.837, SHELF_LIFT_HEIGHT_M))
        self.assertFalse(lifted_box_clears_upper_board(0.837, 0.16))

    def test_side_view_top_hit_snaps_to_l2_center(self):
        snapped = snap_l2_box_center([-2.588, 0.890, 0.931])
        self.assertAlmostEqual(snapped[2], L2_BOX_CENTER_Z)
        np.testing.assert_allclose(snapped[:2], [-2.588, 0.890])
        self.assertFalse(lifted_box_clears_upper_board(0.931, SHELF_LIFT_HEIGHT_M))
        self.assertTrue(lifted_box_clears_upper_board(snapped[2], SHELF_LIFT_HEIGHT_M))

    def test_in_shelf_hug_is_treated_as_resume(self):
        pose = [-1.926, 0.788, -3.074]
        hug_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        hug_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        self.assertTrue(already_holding_on_shelf(pose, 0.366, hug_left, hug_right))
        pregrasp_left = np.array([0.0, 0.0, 0.0, -1.51, -0.766, 1.57])
        pregrasp_right = np.array([0.0, 0.0, 0.0, 1.51, 0.766, -1.57])
        self.assertFalse(already_holding_on_shelf(pose, 0.366, pregrasp_left, pregrasp_right))
        self.assertFalse(already_holding_on_shelf([-1.46, 0.89, math.pi], 0.02, hug_left, hug_right))
        aisle = [-1.664, 0.807, 1.663]
        self.assertFalse(already_holding_on_shelf(aisle, 0.366, hug_left, hug_right))
        self.assertTrue(already_carrying_box(0.366, hug_left, hug_right))
        self.assertTrue(already_at_table_place([-1.06, 1.57, 1.50]))
        self.assertFalse(already_at_table_place([-1.67, 0.81, -3.07]))

    def test_settled_squeeze_is_refreshed_inside_carry_band(self):
        reached_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        reached_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        old_left, old_right = local_carry_hold(reached_left, reached_right)
        settled_left = reached_left + 0.65 * (old_left - reached_left)
        settled_right = reached_right + 0.65 * (old_right - reached_right)
        lost = carry_hold_ok(settled_left, settled_right, old_left, old_right)
        self.assertFalse(lost["holding"])
        new_left, new_right, contact = maintain_carry_hold(settled_left, settled_right, old_left, old_right)
        self.assertTrue(contact["hold_refreshed"])
        self.assertTrue(contact["holding"])
        self.assertGreaterEqual(contact["left_max_joint_residual_rad"], 0.015)
        refreshed = carry_hold_ok(settled_left, settled_right, new_left, new_right)
        self.assertTrue(refreshed["holding"])

    def test_table_place_stand_puts_held_center_on_slot(self):
        held = np.array([0.67, -0.02, 0.917])
        stand = place_stand_xy(INSTRUCTION_PLACE_WORLD, PLACE_YAW, held)
        cosine, sine = math.cos(PLACE_YAW), math.sin(PLACE_YAW)
        world = np.array([
            stand[0] + cosine * held[0] - sine * held[1],
            stand[1] + sine * held[0] + cosine * held[1],
        ])
        np.testing.assert_allclose(world, INSTRUCTION_PLACE_WORLD[:2], atol=1e-12)

    def test_table_scoring_radius_accepts_the_instruction_slot(self):
        inside = table_placement_error([-1.05, 2.18, 0.834], INSTRUCTION_PLACE_WORLD, 0.28)
        self.assertTrue(inside["within_radius"])
        outside = table_placement_error([-1.00, 1.70, 0.834], INSTRUCTION_PLACE_WORLD, 0.28)
        self.assertFalse(outside["within_radius"])

    def test_held_box_over_table_slot_is_ready_to_lower(self):
        held = np.array([0.56, -0.02, 0.834])
        stand = place_stand_xy(INSTRUCTION_PLACE_WORLD, PLACE_YAW, held)
        pose = np.array([stand[0], stand[1], PLACE_YAW])
        ready = box_inside_table_radius(pose, held, INSTRUCTION_PLACE_WORLD, 0.18)
        self.assertTrue(ready["within_radius"])
        self.assertLess(ready["xy_error_m"], 1e-9)

    def test_stale_palm_fk_is_not_trusted_off_the_table(self):
        self.assertFalse(box_on_table_top([-0.99, 1.85, 0.834]))
        self.assertTrue(box_on_table_top([-0.99, 2.10, 0.834]))
        raw = np.array([0.789, -0.055, 0.934])
        clamped = table_place_held_center(raw)
        self.assertAlmostEqual(clamped[0], TABLE_PLACE_HELD_X_MAX)
        stand_raw = place_stand_xy(INSTRUCTION_PLACE_WORLD, PLACE_YAW, raw)
        stand_close = place_stand_xy(INSTRUCTION_PLACE_WORLD, PLACE_YAW, clamped)
        self.assertGreater(stand_close[1], stand_raw[1])

    def test_stalled_shelf_hug_is_locked_instead_of_opened(self):
        pregrasp_left = np.array([0.0, 0.0, 0.0, -1.51, -0.766, 1.57])
        pregrasp_right = np.array([0.0, 0.0, 0.0, 1.51, 0.766, -1.57])
        plan_left = np.array([0.00435, -1.47727, 1.24247, -1.51701, -1.00067, 1.58178])
        plan_right = np.array([-0.00513, -1.24511, 1.16259, 1.50971, 0.84868, -1.56988])
        stalled_left = np.array([0.00845, -1.30698, 1.13170, -1.52272, -0.99691, 1.57523])
        stalled_right = np.array([-0.00568, -1.09771, 1.06204, 1.51653, 0.86175, -1.56188])
        locked = blocked_hug_lock(
            stalled_left, stalled_right, pregrasp_left, pregrasp_right, plan_left, plan_right,
        )
        self.assertIsNotNone(locked)
        self.assertTrue(locked["feedback"]["blocked_hug"])
        np.testing.assert_allclose(locked["left"], stalled_left)
        self.assertIsNone(blocked_hug_lock(
            pregrasp_left, pregrasp_right, pregrasp_left, pregrasp_right, plan_left, plan_right,
        ))

    def test_blocked_hug_hold_stays_in_carry_band(self):
        stalled_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        stalled_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        tight_left = np.array([0.00418, -1.54066, 1.27610, -1.51813, -1.03038, 1.58383])
        tight_right = np.array([-0.00489, -1.30383, 1.17708, 1.51215, 0.89282, -1.57364])
        hold_left, hold_right = inward_hold_from_blocked(stalled_left, stalled_right, tight_left, tight_right)
        residual = carry_hold_ok(stalled_left, stalled_right, hold_left, hold_right)
        self.assertTrue(residual["holding"])
        empty = carry_hold_ok(hold_left, hold_right, hold_left, hold_right)
        self.assertFalse(empty["holding"])

    def test_far_tight_plan_is_capped_inside_carry_band(self):
        stalled_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        stalled_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        far_left = stalled_left + np.array([0.04, -0.90, 0.50, 0.03, 0.03, -0.04])
        far_right = stalled_right + np.array([-0.06, -0.50, 0.40, -0.09, -0.07, 0.12])
        hold_left, hold_right = inward_hold_from_blocked(stalled_left, stalled_right, far_left, far_right)
        residual = carry_hold_ok(stalled_left, stalled_right, hold_left, hold_right)
        self.assertTrue(residual["holding"])
        self.assertLessEqual(residual["left_max_joint_residual_rad"], 0.08)
        self.assertLessEqual(residual["right_max_joint_residual_rad"], 0.08)

    def test_resume_hold_uses_local_j2_squeeze(self):
        stalled_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        stalled_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        hold_left, hold_right = local_carry_hold(stalled_left, stalled_right)
        residual = carry_hold_ok(stalled_left, stalled_right, hold_left, hold_right)
        self.assertTrue(residual["holding"])
        self.assertAlmostEqual(residual["left_max_joint_residual_rad"], 0.04, places=6)


if __name__ == "__main__":
    unittest.main()
