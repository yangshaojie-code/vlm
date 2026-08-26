import inspect
import math
import unittest

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0
from head_camera_kinematics import SLIDE_LIMITS
from task1_pick_lift_check import (
    TASK1_APPROACH_HALF_M,
    TASK1_GRASP_FWD_OFFSET_M,
    TASK1_GRASP_Z_OFFSET_M,
    TASK1_HOLD_HALF_M,
    contact_approach_geometry,
    contact_clearance_schedule,
    lift_slide_target,
)
from task3_cube_top_shelf_place_check import (
    CUBE_TOP_CENTER_Z,
    CUBE_TOP_Z,
    GRASP_YAW,
    INSTRUCTION_PLACE_WORLD,
    PLACE_ACCEPT_RADIUS_M,
    PLACE_RADIUS_M,
    PLACE_YAW,
    TASK3_LIFT_HEIGHT_M,
    TASK3_STANDOFF_M,
    YELLOW_FIXED_WORLD,
    _establish_cube_top_hold,
    box_inside_place_radius,
    center_from_cube_surface,
    lift_clears_cube,
    place_stand_from_goal,
    shelf_inward_ok,
    snap_cube_top_center,
    station_for_yellow,
    task3_placement_error,
    validate_place_world_l1,
    validate_yellow_world,
)


class Task3CubeTopShelfPlaceCheckTests(unittest.TestCase):
    def test_fixed_yellow_is_in_cube_top_window(self):
        np.testing.assert_allclose(validate_yellow_world(YELLOW_FIXED_WORLD), YELLOW_FIXED_WORLD)
        with self.assertRaises(ValueError):
            validate_yellow_world([-2.63, 0.778, 0.837])
        with self.assertRaises(ValueError):
            validate_yellow_world([-0.54, 2.30, 0.834])

    def test_cube_top_center_z_matches_layout(self):
        self.assertAlmostEqual(CUBE_TOP_CENTER_Z, 1.004)
        self.assertAlmostEqual(CUBE_TOP_Z, 0.909)

    def test_side_view_hit_maps_and_snaps_to_cube_top_center(self):
        center = center_from_cube_surface([-0.54, 2.22, 1.03], GRASP_YAW)
        np.testing.assert_allclose(center, [-0.54, 2.30, 1.004], atol=1e-12)
        snapped = snap_cube_top_center([-0.52, 2.33, 1.06])
        np.testing.assert_allclose(snapped, YELLOW_FIXED_WORLD, atol=1e-12)
        north_biased = snap_cube_top_center([-0.621, 2.426, 1.05])
        np.testing.assert_allclose(north_biased, YELLOW_FIXED_WORLD, atol=1e-12)
        stand = station_for_yellow(north_biased, TASK3_STANDOFF_M, GRASP_YAW)
        np.testing.assert_allclose(stand, [-0.54, 1.76, math.pi / 2.0], atol=1e-12)

    def test_fixed_l1_place_world_is_valid(self):
        np.testing.assert_allclose(
            validate_place_world_l1(INSTRUCTION_PLACE_WORLD), INSTRUCTION_PLACE_WORLD,
        )
        with self.assertRaises(ValueError):
            validate_place_world_l1([-2.68, 0.778, 1.156])
        with self.assertRaises(ValueError):
            validate_place_world_l1([-1.00, 2.20, 0.834])

    def test_lift_clears_cube_with_margin(self):
        self.assertTrue(lift_clears_cube(YELLOW_FIXED_WORLD[2], TASK3_LIFT_HEIGHT_M))
        self.assertFalse(lift_clears_cube(YELLOW_FIXED_WORLD[2], 0.03))

    def test_station_is_south_of_table_edge(self):
        stand = station_for_yellow(YELLOW_FIXED_WORLD, TASK3_STANDOFF_M, GRASP_YAW)
        np.testing.assert_allclose(stand, [-0.54, 1.76, math.pi / 2.0], atol=1e-12)
        self.assertLess(stand[1], 1.885)
        barely_over = station_for_yellow([-0.54, 2.4256, 1.004], TASK3_STANDOFF_M, GRASP_YAW)
        self.assertAlmostEqual(barely_over[1], 1.885)
        with self.assertRaises(ValueError):
            station_for_yellow([-0.54, 2.62, 1.004], TASK3_STANDOFF_M, GRASP_YAW)

    def test_hug_targets_use_verified_task1_geometry_at_cube_top_height(self):
        box_base = np.array([0.56, -0.01, 1.004])
        left, right = contact_approach_geometry(
            box_base, 0.0, TASK1_GRASP_FWD_OFFSET_M, TASK1_GRASP_Z_OFFSET_M, TASK1_HOLD_HALF_M,
        )
        np.testing.assert_allclose(left, [0.625, TASK1_HOLD_HALF_M - 0.01, 1.049], atol=1e-12)
        np.testing.assert_allclose(right, [0.625, -TASK1_HOLD_HALF_M - 0.01, 1.049], atol=1e-12)
        self.assertGreater(left[2], CUBE_TOP_Z, "palms must hug above the cube top")

    def test_fixed_layout_slide_chain_stays_inside_limits(self):
        contact_slide = float(PRE_GRASP_Z0 - (YELLOW_FIXED_WORLD[2] + TASK1_GRASP_Z_OFFSET_M))
        lift_slide = lift_slide_target(contact_slide, TASK3_LIFT_HEIGHT_M)
        held_z = YELLOW_FIXED_WORLD[2] + TASK3_LIFT_HEIGHT_M
        clearance_slide = lift_slide + (held_z - (INSTRUCTION_PLACE_WORLD[2] + 0.055))
        place_slide = clearance_slide + 0.055
        for slide in (contact_slide, lift_slide, clearance_slide, place_slide):
            self.assertTrue(SLIDE_LIMITS[0] <= slide <= SLIDE_LIMITS[1], msg=f"slide {slide} out of range")
        self.assertLessEqual(place_slide, SLIDE_LIMITS[1] - 0.02, "place slide needs margin at the L1 depth")

    def test_contact_clearance_schedule_ends_at_zero(self):
        self.assertEqual(contact_clearance_schedule(0.02, 0.01), [0.02, 0.01, 0.0])

    def test_place_stand_puts_held_box_on_place_point(self):
        held = np.array([0.56, -0.01, 1.10])
        stand = place_stand_from_goal(INSTRUCTION_PLACE_WORLD, PLACE_YAW, held)
        cosine, sine = math.cos(PLACE_YAW), math.sin(PLACE_YAW)
        held_world = np.array([
            stand[0] + cosine * held[0] - sine * held[1],
            stand[1] + sine * held[0] + cosine * held[1],
        ])
        np.testing.assert_allclose(held_world, INSTRUCTION_PLACE_WORLD[:2], atol=1e-12)

    def test_place_stand_rejects_unsafe_held_reach(self):
        with self.assertRaises(ValueError):
            place_stand_from_goal(INSTRUCTION_PLACE_WORLD, PLACE_YAW, np.array([0.95, 0.0, 1.10]))

    def test_box_inside_place_radius_boundaries(self):
        base = np.array([-2.12, 0.54, math.pi])
        inside = box_inside_place_radius(base, np.array([0.56, 0.0, 0.55]), INSTRUCTION_PLACE_WORLD, PLACE_ACCEPT_RADIUS_M)
        self.assertTrue(inside["within_radius"])
        self.assertLessEqual(inside["xy_error_m"], PLACE_ACCEPT_RADIUS_M)
        far = box_inside_place_radius(base, np.array([0.56, 0.20, 0.55]), INSTRUCTION_PLACE_WORLD, PLACE_ACCEPT_RADIUS_M)
        self.assertFalse(far["within_radius"])

    def test_task3_placement_error_radius_and_z(self):
        held = np.array([-2.68, 0.54, 0.498])
        error = task3_placement_error(held, INSTRUCTION_PLACE_WORLD, PLACE_RADIUS_M)
        self.assertTrue(error["within_radius"])
        self.assertAlmostEqual(error["xy_error_m"], 0.0)
        self.assertAlmostEqual(error["z_error_m"], 0.0)
        outside = task3_placement_error(held + np.array([0.25, 0.0, 0.0]), INSTRUCTION_PLACE_WORLD, PLACE_RADIUS_M)
        self.assertFalse(outside["within_radius"])

    def test_shelf_inward_ok_flags_lip_hang(self):
        deep = shelf_inward_ok(np.array([-2.72, 0.54, 0.498]), INSTRUCTION_PLACE_WORLD)
        self.assertTrue(deep["deep_enough"])
        hanging = shelf_inward_ok(np.array([-2.55, 0.54, 0.498]), INSTRUCTION_PLACE_WORLD)
        self.assertFalse(hanging["deep_enough"])
        self.assertGreater(hanging["outward_m"], 0.10)

    def test_yellow_hug_window_from_station(self):
        box_base = np.array([0.56, -0.01, 1.004])
        self.assertTrue(0.35 <= box_base[0] <= 0.75)
        self.assertLessEqual(abs(box_base[1]), 0.15)

    def test_cube_top_hold_lifts_before_leaving_the_table(self):
        source = inspect.getsource(_establish_cube_top_hold)
        lift_at = source.find("_traverse_spine_holding")
        self.assertGreaterEqual(lift_at, 0)
        self.assertIn("lift_slide", source[lift_at:lift_at + 200])
        self.assertLess(source.find("held_center_base"), source.find("return context"))


if __name__ == "__main__":
    unittest.main()
