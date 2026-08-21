import math
import unittest

import numpy as np

from task1_shelf_place_check import (
    INSTRUCTION_PLACE_WORLD,
    PLACE_YAW,
    STAGING_BACK_M,
    TABLE_LEAVE_MIN_TRAVELED_M,
    apply_slide_keep_hold,
    box_inside_place_radius,
    held_center_world,
    held_line_command,
    load_hold_resume,
    place_stand_from_goal,
    placement_error,
    pose_offset,
    release_cartesian,
    slide_for_held_z,
    staging_pose,
    validate_place_world,
)


class Task1ShelfPlaceCheckTests(unittest.TestCase):
    def test_instruction_place_world_is_accepted(self):
        np.testing.assert_allclose(validate_place_world(INSTRUCTION_PLACE_WORLD), INSTRUCTION_PLACE_WORLD)
        with self.assertRaises(ValueError):
            validate_place_world([-1.00, 2.20, 0.834])

    def test_place_stand_puts_held_center_on_place_world_xy(self):
        held = np.array([0.5641200671432587, -0.022416735201684492, 0.9324331344648043])
        stand = place_stand_from_goal(INSTRUCTION_PLACE_WORLD, PLACE_YAW, held)
        world = held_center_world([stand[0], stand[1], PLACE_YAW], held)
        np.testing.assert_allclose(world[:2], INSTRUCTION_PLACE_WORLD[:2], atol=1e-12)

    def test_staging_pose_is_half_meter_behind_west_place_stand(self):
        stand = np.array([-2.11588, 0.755583])
        pose = staging_pose(stand, PLACE_YAW, STAGING_BACK_M)
        np.testing.assert_allclose(pose[:2], [-1.61588, 0.755583], atol=1e-12)
        self.assertAlmostEqual(abs(pose[2]), math.pi, places=12)

    def test_west_retreat_increases_x_without_changing_yaw(self):
        start = np.array([-2.12, 0.76, PLACE_YAW])
        target = pose_offset(start, 0.32, reverse=True)
        np.testing.assert_allclose(target, [-1.80, 0.76, PLACE_YAW], atol=1e-12)

    def test_slide_keep_hold_raises_held_z_when_slide_decreases(self):
        held = apply_slide_keep_hold([0.56, -0.02, 0.832], 0.444, 0.344)
        np.testing.assert_allclose(held, [0.56, -0.02, 0.932], atol=1e-12)
        target = slide_for_held_z(0.344, held[2], 1.156 + 0.055)
        self.assertAlmostEqual(target, 0.065, places=12)
        lowered = slide_for_held_z(target, apply_slide_keep_hold(held, 0.344, target)[2], 1.156)
        self.assertAlmostEqual(lowered, 0.120, places=12)

    def test_release_spreads_laterally_only(self):
        left, right = release_cartesian([0.63, 0.12, 1.20], [0.63, -0.16, 1.20], 0.04)
        np.testing.assert_allclose(left, [0.63, 0.16, 1.20], atol=1e-12)
        np.testing.assert_allclose(right, [0.63, -0.20, 1.20], atol=1e-12)

    def test_placement_error_uses_instruction_radius(self):
        inside = placement_error([-2.70, 0.80, 1.16], INSTRUCTION_PLACE_WORLD, 0.24)
        self.assertTrue(inside["within_radius"])
        self.assertLess(inside["xy_error_m"], 0.24)
        outside = placement_error([-2.20, 0.778, 1.156], INSTRUCTION_PLACE_WORLD, 0.24)
        self.assertFalse(outside["within_radius"])

    def test_table_leave_does_not_stop_at_p2_early_complete(self):
        start = np.array([-1.010, 1.684, math.pi / 2.0])
        target = pose_offset(start, 0.22, reverse=True)
        early = start.copy()
        early[1] -= 0.170
        linear, angular, details = held_line_command(
            early, start, target, -1, 0.03, 0.05, min_traveled_m=TABLE_LEAVE_MIN_TRAVELED_M, max_linear_speed=0.08,
        )
        self.assertEqual(details["phase"], "translate")
        self.assertLess(linear, 0.0)
        self.assertAlmostEqual(details["traveled_m"], 0.170, places=12)

        done = start.copy()
        done[1] -= 0.205
        linear, angular, details = held_line_command(
            done, start, target, -1, 0.02, 0.05, min_traveled_m=TABLE_LEAVE_MIN_TRAVELED_M, max_linear_speed=0.08,
        )
        self.assertEqual((linear, angular, details["phase"]), (0.0, 0.0, "complete"))
        self.assertGreaterEqual(details["traveled_m"], TABLE_LEAVE_MIN_TRAVELED_M)

    def test_hold_resume_accepts_failed_staging_report(self):
        report = {
            "mode": "task1_bimanual_hug_shelf_place_check",
            "phase": "staging_nav",
            "lift_completed": True,
            "table_leave_completed": True,
            "hold_joint_targets": {"left": [0.0] * 6, "right": [0.1] * 6},
            "held_center_base_after_lift": [0.56, -0.02, 0.93],
            "staging_pose": [-1.62, 0.76, -math.pi],
            "place_world": INSTRUCTION_PLACE_WORLD.tolist(),
            "initial_slide": 0.0,
            "contact_slide": 0.44,
            "lift_slide": 0.34,
        }
        import json
        from unittest.mock import patch
        with patch("task1_shelf_place_check.Path.read_text", return_value=json.dumps(report)):
            resume = load_hold_resume("failed.json")
        self.assertEqual(resume["phase"], "staging_nav")
        np.testing.assert_allclose(resume["held_center_base"], [0.56, -0.02, 0.93], atol=1e-12)
        with patch("task1_shelf_place_check.Path.read_text", return_value=json.dumps({**report, "phase": "hug_lift"})):
            with self.assertRaisesRegex(ValueError, "cannot continue"):
                load_hold_resume("failed.json")

    def test_stuck_in_cabinet_is_ready_to_lower(self):
        inside_score = box_inside_place_radius(
            [-1.950570715258924, 0.7742018429989654, -3.0733755391367796],
            [0.563357774258634, -0.017686131954093007, 1.2109999999999999],
            INSTRUCTION_PLACE_WORLD,
            0.24,
        )
        ready = box_inside_place_radius(
            [-1.950570715258924, 0.7742018429989654, -3.0733755391367796],
            [0.563357774258634, -0.017686131954093007, 1.2109999999999999],
            INSTRUCTION_PLACE_WORLD,
            0.18,
        )
        self.assertTrue(inside_score["within_radius"])
        self.assertTrue(ready["within_radius"])

    def test_front_of_cabinet_is_inside_score_radius_but_not_ready_to_release(self):
        air = box_inside_place_radius(
            [-1.878842279902686, 0.7791064455127821, -3.0733873796159137],
            [0.563357774258634, -0.017686131954093007, 1.2109999999999999],
            INSTRUCTION_PLACE_WORLD,
            0.24,
        )
        ready = box_inside_place_radius(
            [-1.878842279902686, 0.7791064455127821, -3.0733873796159137],
            [0.563357774258634, -0.017686131954093007, 1.2109999999999999],
            INSTRUCTION_PLACE_WORLD,
            0.18,
        )
        self.assertTrue(air["within_radius"])
        self.assertFalse(ready["within_radius"])

    def test_hold_resume_skips_to_approach_after_raise(self):
        report = {
            "mode": "task1_bimanual_hug_shelf_place_check",
            "phase": "shelf_approach",
            "lift_completed": True,
            "table_leave_completed": True,
            "shelf_raise_completed": True,
            "hold_joint_targets": {"left": [0.0] * 6, "right": [0.1] * 6},
            "held_center_base_after_lift": [0.56, -0.02, 0.93],
            "held_center_base_at_clearance": [0.56, -0.02, 1.211],
            "staging_pose": [-1.62, 0.76, -math.pi],
            "place_world": INSTRUCTION_PLACE_WORLD.tolist(),
            "initial_slide": 0.0,
            "contact_slide": 0.44,
            "lift_slide": 0.34,
        }
        import json
        from unittest.mock import patch
        with patch("task1_shelf_place_check.Path.read_text", return_value=json.dumps(report)):
            resume = load_hold_resume("failed.json")
        self.assertTrue(resume["skip_to_approach"])
        np.testing.assert_allclose(resume["held_center_base"], [0.56, -0.02, 1.211], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
