import inspect
import math
import unittest

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0
from head_camera_kinematics import SLIDE_LIMITS
from task1_pick_lift_check import (
    TASK1_GRASP_FWD_OFFSET_M,
    TASK1_HOLD_HALF_M,
    contact_approach_geometry,
    contact_clearance_schedule,
    lift_slide_target,
)
from task3_cube_top_shelf_place_check import (
    APPROACH_STALL_X_M,
    BOX_HALF_DEPTH_M,
    CUBE_TOP_CENTER_Z,
    CUBE_TOP_Z,
    GRASP_YAW,
    INSTRUCTION_PLACE_WORLD,
    PLACE_ACCEPT_RADIUS_M,
    PLACE_RADIUS_M,
    PLACE_YAW,
    PLACE_ALIGN_YAW_TOLERANCE_RAD,
    PLACE_INSERT_YAW_TOLERANCE_RAD,
    TASK3_MAX_PLACE_OUTWARD_M,
    TASK3_PLACE_REMAINING_OK_M,
    SHELF_AISLE_Y_M,
    SHELF_POST_HALF_M,
    SHELF_POST_LOCAL_Y_M,
    STATION_Y_MAX,
    PACKAGING_WORLD,
    TASK3_GRASP_FWD_OFFSET_M,
    TASK3_GRASP_Z_OFFSET_M,
    TASK3_HOLD_HALF_M,
    TASK3_HOLD_LINEAR_SPEED,
    TASK3_HOLD_SQUEEZE_RAD,
    TASK3_LIFT_HEIGHT_M,
    TASK3_RELEASE_OPEN_RAD,
    TASK3_RELEASE_SPREAD_M,
    TASK3_RELEASE_WITHDRAW_M,
    TASK3_SHELF_LINEAR_SPEED,
    TASK3_TABLE_LEAVE_LINEAR_SPEED,
    TASK3_STANDOFF_M,
    YELLOW_FIXED_WORLD,
    _bind_capped_hold_keeper,
    _establish_cube_top_hold,
    _recover,
    aisle_staging_from_stand,
    aligned_shelf_insert_plan,
    approach_bearing,
    approach_clears_packaging,
    box_inside_place_radius,
    center_from_cube_surface,
    cubby_fits_l1,
    hold_palm_metrics,
    insert_line_y_at_x,
    l1_clear_bay_y,
    l1_release_cartesian,
    level_hold_pose,
    local_release_open,
    lift_clears_cube,
    main,
    nearest_shelf_board_z,
    place_is_l1_layer,
    place_left_of_obstacle,
    place_stand_from_goal,
    shelf_inward_ok,
    snap_cube_top_center,
    snap_packaging_center,
    south_then_west_insert_plan,
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
        self.assertAlmostEqual(snapped[0], -0.54)
        self.assertAlmostEqual(snapped[2], 1.004)
        self.assertAlmostEqual(snapped[1], 2.31, places=6)
        north_biased = snap_cube_top_center([-0.621, 2.426, 1.05])
        self.assertAlmostEqual(north_biased[0], -0.54)
        self.assertAlmostEqual(north_biased[2], 1.004)
        self.assertLessEqual(north_biased[1], STATION_Y_MAX + TASK3_STANDOFF_M - 0.01)
        stand = station_for_yellow(north_biased, TASK3_STANDOFF_M, GRASP_YAW)
        self.assertLessEqual(stand[1], STATION_Y_MAX)

    def test_south_biased_detection_stays_near_the_visual_center(self):
        hug = snap_cube_top_center([-0.545, 2.254, 1.004])
        self.assertAlmostEqual(hug[0], -0.54)
        self.assertAlmostEqual(hug[2], 1.004)
        self.assertAlmostEqual(hug[1], 2.274, places=6)
        self.assertLess(abs(hug[1] - 2.254), 0.021)
        self.assertGreater(YELLOW_FIXED_WORLD[1] - hug[1], 0.02)
        stand = station_for_yellow(hug, TASK3_STANDOFF_M, GRASP_YAW)
        self.assertAlmostEqual(stand[1], hug[1] - TASK3_STANDOFF_M)

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
        np.testing.assert_allclose(stand, [-0.54, 1.80, math.pi / 2.0], atol=1e-12)
        self.assertLess(stand[1], 1.885)
        barely_over = station_for_yellow([-0.54, 2.39, 1.004], TASK3_STANDOFF_M, GRASP_YAW)
        self.assertAlmostEqual(barely_over[1], STATION_Y_MAX)
        with self.assertRaises(ValueError):
            station_for_yellow([-0.54, 2.62, 1.004], TASK3_STANDOFF_M, GRASP_YAW)

    def test_hug_covers_the_side_face_center_not_the_far_lip(self):
        box_base = np.array([0.56, -0.01, 1.004])
        left, right = contact_approach_geometry(
            box_base, 0.0, TASK3_GRASP_FWD_OFFSET_M, TASK3_GRASP_Z_OFFSET_M, TASK1_HOLD_HALF_M,
        )
        np.testing.assert_allclose(
            left, [0.56 + TASK3_GRASP_FWD_OFFSET_M, TASK1_HOLD_HALF_M - 0.01, 1.004], atol=1e-12,
        )
        np.testing.assert_allclose(
            right, [0.56 + TASK3_GRASP_FWD_OFFSET_M, -TASK1_HOLD_HALF_M - 0.01, 1.004], atol=1e-12,
        )
        self.assertGreater(left[2], CUBE_TOP_Z, "palms must hug above the cube top")
        self.assertGreater(TASK3_GRASP_FWD_OFFSET_M, 0.04)
        self.assertLess(TASK3_GRASP_FWD_OFFSET_M, BOX_HALF_DEPTH_M)
        far_strip = BOX_HALF_DEPTH_M - TASK3_GRASP_FWD_OFFSET_M
        self.assertGreater(far_strip, 0.01)

    def test_fixed_layout_slide_chain_stays_inside_limits(self):
        from task3_cube_top_shelf_place_check import TASK3_APPROACH_Z_M, SHELF_L2_BOARD_Z, held_bottom_z, held_top_z
        contact_slide = float(PRE_GRASP_Z0 - (YELLOW_FIXED_WORLD[2] + TASK3_GRASP_Z_OFFSET_M))
        lift_slide = lift_slide_target(contact_slide, TASK3_LIFT_HEIGHT_M)
        held_z = YELLOW_FIXED_WORLD[2] + TASK3_LIFT_HEIGHT_M
        approach_slide = lift_slide + (held_z - TASK3_APPROACH_Z_M)
        place_slide = approach_slide + (TASK3_APPROACH_Z_M - INSTRUCTION_PLACE_WORLD[2])
        for slide in (contact_slide, lift_slide, approach_slide, place_slide):
            self.assertTrue(SLIDE_LIMITS[0] <= slide <= SLIDE_LIMITS[1], msg=f"slide {slide} out of range")
        self.assertLessEqual(place_slide, SLIDE_LIMITS[1] - 0.02, "place slide needs margin at the L1 depth")
        self.assertGreater(TASK3_APPROACH_Z_M, 0.52)
        self.assertLess(TASK3_APPROACH_Z_M, 0.62)
        self.assertTrue(cubby_fits_l1(TASK3_APPROACH_Z_M))
        self.assertFalse(approach_clears_packaging(TASK3_APPROACH_Z_M))
        self.assertTrue(place_is_l1_layer(INSTRUCTION_PLACE_WORLD[2]))
        self.assertLessEqual(held_top_z(TASK3_APPROACH_Z_M), SHELF_L2_BOARD_Z - 0.04)
        self.assertGreater(held_bottom_z(TASK3_APPROACH_Z_M), 0.45)

    def test_place_left_of_detected_packaging_is_same_layer_south(self):
        derived = place_left_of_obstacle(PACKAGING_WORLD)
        post_inner_y = SHELF_AISLE_Y_M - SHELF_POST_LOCAL_Y_M + SHELF_POST_HALF_M
        packaging_south_y = PACKAGING_WORLD[1] - 0.051
        bay_mid_y = l1_clear_bay_y(PACKAGING_WORLD[1])
        self.assertAlmostEqual(derived[0], INSTRUCTION_PLACE_WORLD[0], places=2)
        self.assertAlmostEqual(bay_mid_y, 0.5 * (post_inner_y + packaging_south_y))
        self.assertAlmostEqual(derived[1], bay_mid_y)
        self.assertGreater(derived[1], post_inner_y)
        self.assertLess(derived[1], packaging_south_y)
        self.assertLess(
            abs(derived[1] - INSTRUCTION_PLACE_WORLD[1]),
            PLACE_RADIUS_M,
        )
        self.assertAlmostEqual(derived[2], INSTRUCTION_PLACE_WORLD[2], places=2)
        self.assertAlmostEqual(nearest_shelf_board_z(PACKAGING_WORLD[2]), 0.403)
        self.assertAlmostEqual(TASK3_HOLD_HALF_M, 0.115)
        self.assertGreater(TASK3_HOLD_SQUEEZE_RAD, 0.04)
        np.testing.assert_allclose(snap_packaging_center([-2.55, 0.80, 0.55]), PACKAGING_WORLD)

    def test_apply_detects_obstacle_then_squeezes_from_contact(self):
        source = inspect.getsource(main)
        self.assertIn("locate_packaging", source)
        self.assertIn("place_left_of_obstacle", source)
        self.assertIn("layout_packaging", source)
        self.assertIn("local_carry_hold", source)
        self.assertNotIn("inward_hold_from_blocked", source)

    def test_apply_lowers_to_l1_after_south_before_release(self):
        source = inspect.getsource(main)
        south_at = source.find("shelf_approach_face_south")
        lower_at = source.find('phase = "place_lower"')
        release_at = source.find('phase = "release"')
        self.assertGreater(south_at, 0)
        self.assertGreater(lower_at, south_at)
        self.assertGreater(release_at, lower_at)
        self.assertIn("place_layer_z_m", source)
        self.assertNotIn("pre_lower_estimated_place_world", source)

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
        self.assertAlmostEqual(TASK3_MAX_PLACE_OUTWARD_M, 0.06)
        almost = shelf_inward_ok(np.array([-2.636, 0.578, 0.530]), np.array([-2.68, 0.5685, 0.498]))
        self.assertTrue(almost["deep_enough"])
        almost_xy = task3_placement_error(
            np.array([-2.636, 0.578, 0.530]),
            np.array([-2.68, 0.5685, 0.498]),
            PLACE_ACCEPT_RADIUS_M,
        )
        self.assertTrue(almost_xy["within_radius"])
        self.assertLessEqual(0.047, TASK3_PLACE_REMAINING_OK_M)

    def test_l1_release_opens_wider_than_the_box_then_base_backs_out(self):
        left = np.array([0.01, -0.98, 0.56, -1.53, -1.07, 1.60])
        right = np.array([-0.01, -1.19, 0.59, 1.54, 1.26, -1.61])
        open_left, open_right = local_release_open(left, right)
        self.assertAlmostEqual(open_left[1], left[1] + TASK3_RELEASE_OPEN_RAD)
        self.assertAlmostEqual(open_right[1], right[1] + TASK3_RELEASE_OPEN_RAD)
        self.assertAlmostEqual(TASK3_RELEASE_SPREAD_M, 0.04)
        self.assertAlmostEqual(TASK3_RELEASE_WITHDRAW_M, 0.04)
        last_left = np.array([0.586, 0.098, 0.503])
        last_right = np.array([0.584, -0.080, 0.502])
        too_tight = last_left[1] - last_right[1]
        self.assertLess(too_tight, 2 * 0.12)
        opened_left, opened_right = l1_release_cartesian(last_left, last_right)
        self.assertGreater(opened_left[1] - opened_right[1], 2 * 0.12)
        self.assertAlmostEqual(opened_left[0], last_left[0] - TASK3_RELEASE_WITHDRAW_M)
        self.assertAlmostEqual(opened_right[0], last_right[0] - TASK3_RELEASE_WITHDRAW_M)
        source = inspect.getsource(main)
        release_at = source.find('phase = "release"')
        retreat_at = source.find('phase = "shelf_retreat"')
        retract_at = source.find('phase = "retract"')
        self.assertGreater(release_at, 0)
        self.assertGreater(retreat_at, release_at)
        self.assertGreater(retract_at, retreat_at)
        self.assertIn("l1_release_joints", source)
        self.assertIn("recovery_released_on_shelf", inspect.getsource(_recover))
        self.assertGreaterEqual(TASK3_HOLD_LINEAR_SPEED, 0.18)
        self.assertGreaterEqual(TASK3_SHELF_LINEAR_SPEED, 0.12)
        self.assertGreaterEqual(TASK3_TABLE_LEAVE_LINEAR_SPEED, 0.16)

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

    def test_cube_top_hold_squeezes_inward_instead_of_locking_open_gap(self):
        source = inspect.getsource(_establish_cube_top_hold)
        squeeze_at = source.find("local_carry_hold")
        open_lock = source.find('hold_left = np.asarray(plan["left_joint_target"]')
        self.assertGreaterEqual(squeeze_at, 0)
        self.assertEqual(open_lock, -1)
        self.assertIn("close_to_hold_gap", source)
        self.assertIn("level_hold_pose", source)
        self.assertIn("TASK3_HOLD_SQUEEZE_RAD", source)
        self.assertIn("MAX_HOLD_PALM_DZ_M", source)
        self.assertIn("hug_center_error_m", source)

    def test_open_corner_graze_is_not_treated_as_a_side_hold(self):
        slide = 0.2742
        left = np.array([0.00299, -0.92967, 0.57138, -1.52685, -1.12736, 1.59393])
        right = np.array([-0.00377, -1.10212, 0.59878, 1.52954, 1.26854, -1.60072])
        metrics = hold_palm_metrics(slide, left, right)
        self.assertGreater(metrics["half_span_m"], 0.122)
        self.assertFalse(metrics["level_enough"])

    def test_apply_backs_away_before_hugging_from_a_stale_station(self):
        source = inspect.getsource(main)
        self.assertIn("should_backup_to_observe", source)
        self.assertIn("refusing the nominal cube-top", source)

    def test_aisle_insert_clears_the_south_stall_line(self):
        held = np.array([0.57, 0.026, 0.85])
        stand = place_stand_from_goal(INSTRUCTION_PLACE_WORLD, PLACE_YAW, held)
        stage, insert = aisle_staging_from_stand(stand, PLACE_YAW)
        plan = south_then_west_insert_plan(stage[:2], stand, place_yaw=PLACE_YAW)
        self.assertAlmostEqual(insert[1], SHELF_AISLE_Y_M)
        self.assertAlmostEqual(stage[1], SHELF_AISLE_Y_M)
        self.assertGreater(stage[0], insert[0])
        np.testing.assert_allclose(plan["south_xy"], [stage[0], stand[1]])
        self.assertGreater(plan["south_xy"][0], APPROACH_STALL_X_M)
        due_west_y = insert_line_y_at_x([-1.58, 0.59], stand, APPROACH_STALL_X_M)
        self.assertLess(due_west_y, 0.62)
        self.assertTrue(plan["needs_south_shift"])
        self.assertAlmostEqual(abs(plan["west_yaw"]), math.pi)
        self.assertAlmostEqual(plan["south_bearing"], -math.pi / 2, places=2)
        self.assertGreater(TASK3_SHELF_LINEAR_SPEED, 0.12)
        self.assertLess(PLACE_ALIGN_YAW_TOLERANCE_RAD, 0.08)
        self.assertLess(PLACE_INSERT_YAW_TOLERANCE_RAD, PLACE_ALIGN_YAW_TOLERANCE_RAD)

    def test_apply_uses_south_then_west_insert(self):
        source = inspect.getsource(main)
        self.assertIn("south_then_west_insert_plan", source)
        self.assertIn("shelf_approach_south", source)
        self.assertIn("shelf_approach_west", source)
        self.assertIn("shelf_approach_square_west", source)
        self.assertIn("PLACE_INSERT_YAW_TOLERANCE_RAD", source)
        self.assertIn("l1_clear_bay_y", inspect.getsource(place_left_of_obstacle))
        self.assertNotIn("l1_pole_clear_hold_plan", source)
        self.assertNotIn("l1_south_arm_tuck", source)
        self.assertNotIn("shelf_approach_west_then_south", source)

    def test_tight_confirmed_squeeze_is_still_a_side_hold(self):
        slide = 0.2742
        sagged_left = np.array([0.01313, -0.97776, 0.56319, -1.53468, -1.06688, 1.59804])
        sagged_right = np.array([-0.01792, -1.19683, 0.59025, 1.54584, 1.26125, -1.61089])
        before = hold_palm_metrics(slide, sagged_left, sagged_right)
        self.assertGreaterEqual(before["half_span_m"], 0.075)
        self.assertLess(before["half_span_m"], 0.10)
        self.assertLessEqual(abs(before["dz_m"]), 0.008)
        self.assertTrue(before["level_enough"])

    def test_level_hold_equalizes_palm_height_without_narrowing_past_the_side_faces(self):
        slide = 0.2742
        sagged_left = np.array([0.01313, -0.97776, 0.56319, -1.53468, -1.06688, 1.59804])
        sagged_right = np.array([-0.01792, -1.19683, 0.59025, 1.54584, 1.26125, -1.61089])
        before = hold_palm_metrics(slide, sagged_left, sagged_right)
        self.assertLess(before["half_span_m"], 0.10)
        plan = level_hold_pose(slide, sagged_left, sagged_right, TASK1_HOLD_HALF_M)
        after = hold_palm_metrics(
            slide, plan["left_joint_target"], plan["right_joint_target"],
        )
        self.assertLessEqual(abs(after["dz_m"]), 1e-5)
        self.assertAlmostEqual(after["half_span_m"], TASK1_HOLD_HALF_M, places=3)
        self.assertTrue(after["level_enough"])

    def test_cube_top_hold_relevels_only_when_palms_are_uneven(self):
        source = inspect.getsource(_establish_cube_top_hold)
        self.assertIn('abs(result["hold_palms_after_squeeze"]["dz_m"]) > MAX_HOLD_PALM_DZ_M', source)
        self.assertNotIn('if not result["hold_palms_after_squeeze"]["level_enough"]', source)

    def test_capped_hold_keeper_refreshes_at_most_once(self):
        reached_left = np.array([0.00554, -1.29823, 1.13427, -1.52135, -0.99482, 1.57636])
        reached_right = np.array([-0.00902, -1.10025, 1.05938, 1.51400, 0.85419, -1.56986])
        from task2_shelf_to_table_check import local_carry_hold
        old_left, old_right = local_carry_hold(reached_left, reached_right)
        settled_left = reached_left + 0.65 * (old_left - reached_left)
        settled_right = reached_right + 0.65 * (old_right - reached_right)
        context = {}
        result = {}
        keeper = _bind_capped_hold_keeper(context, result, max_refresh=1)
        new_left, new_right, first = keeper(settled_left, settled_right, old_left, old_right)
        self.assertTrue(first.get("hold_refreshed"))
        self.assertEqual(result["hold_refresh_count"], 1)
        locked_left, locked_right, second = keeper(settled_left, settled_right, old_left, old_right)
        self.assertFalse(second.get("hold_refreshed"))
        np.testing.assert_allclose(locked_left, old_left)
        np.testing.assert_allclose(locked_right, old_right)


if __name__ == "__main__":
    unittest.main()
