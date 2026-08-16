import unittest
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from task1_precontact_check import (
    center_from_surface,
    load_position_reference,
    navigation_command,
    station_target,
    validate_approach_geometry,
)


class Task1PrecontactTests(unittest.TestCase):
    @staticmethod
    def _odom(x, y, yaw):
        return SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)),
        )))

    def test_station_target_matches_fixed_task1_geometry(self):
        target = station_target([-1.0, 2.2, 0.834])
        np.testing.assert_allclose(target, [-1.0, 1.66, np.pi / 2.0], atol=1e-9)

    def test_center_ray_compensation(self):
        surface = np.array([0.0, 0.08, 0.08])
        center = center_from_surface(surface, 0.095)
        np.testing.assert_allclose(center, [0.0, 0.16, 0.834], atol=1e-12)

    def test_approach_geometry_preserves_clearance_and_symmetry(self):
        left, right = validate_approach_geometry([0.54, 0.0, 0.834], 0.03, -0.05, 0.045)
        np.testing.assert_allclose(left, [0.49, 0.16, 0.879])
        np.testing.assert_allclose(right, [0.49, -0.16, 0.879])

    def test_unsafe_clearance_and_station_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_approach_geometry([0.54, 0.0, 0.834], 0.01, -0.05, 0.045)
        with self.assertRaises(ValueError):
            validate_approach_geometry([0.20, 0.0, 0.834], 0.03, -0.05, 0.045)

    def test_navigation_rotates_before_translating_when_target_is_off_axis(self):
        linear, angular, phase = navigation_command(
            [0.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2.0],
            0.05, 0.05, 0.15, 0.50,
        )
        self.assertEqual(phase, "align_to_target")
        self.assertEqual(linear, 0.0)
        self.assertAlmostEqual(angular, 0.50)

    def test_navigation_translates_with_bounded_velocity(self):
        linear, angular, phase = navigation_command(
            [0.0, 0.0, 0.0], [1.0, 0.05, 0.0],
            0.05, 0.05, 0.15, 0.50,
        )
        self.assertEqual(phase, "translate")
        self.assertGreater(linear, 0.0)
        self.assertLessEqual(linear, 0.15)
        self.assertLessEqual(abs(angular), 0.50)

    def test_navigation_finishes_xy_before_final_yaw(self):
        command = navigation_command(
            [1.01, 1.01, 0.0], [1.0, 1.0, np.pi / 2.0],
            0.05, 0.05, 0.15, 0.50,
        )
        self.assertEqual(command, (0.0, 0.50, "final_yaw"))
        command = navigation_command(
            [1.01, 1.01, np.pi / 2.0 - 0.01], [1.0, 1.0, np.pi / 2.0],
            0.05, 0.05, 0.15, 0.50,
        )
        self.assertEqual(command, (0.0, 0.0, "complete"))

    def test_passed_position_report_is_transformed_using_current_odom(self):
        report = {
            "mode": "task1_pink_precontact_check", "stage": "position", "status": "passed",
            "navigation_phase": "complete", "remaining_position_error_m": 0.04,
            "remaining_yaw_error_rad": -0.04, "final_base": [-1.0, 1.66, np.pi / 2.0],
            "detection": {"center_world": [-1.0, 2.20, 0.834]},
        }
        node = SimpleNamespace(sensors=SimpleNamespace(odom=self._odom(-1.0, 1.66, np.pi / 2.0)))
        with patch("task1_precontact_check.Path.read_text", return_value=json.dumps(report)):
            reference = load_position_reference("position.json", node, 0.05, 0.05)
        np.testing.assert_allclose(reference["center_base"], [0.54, 0.0, 0.834], atol=1e-9)

    def test_position_report_rejects_base_drift(self):
        report = {
            "mode": "task1_pink_precontact_check", "stage": "position", "status": "passed",
            "navigation_phase": "complete", "remaining_position_error_m": 0.04,
            "remaining_yaw_error_rad": 0.04, "final_base": [-1.0, 1.66, np.pi / 2.0],
            "detection": {"center_world": [-1.0, 2.20, 0.834]},
        }
        node = SimpleNamespace(sensors=SimpleNamespace(odom=self._odom(-0.8, 1.66, np.pi / 2.0)))
        with patch("task1_precontact_check.Path.read_text", return_value=json.dumps(report)):
            with self.assertRaisesRegex(RuntimeError, "base moved"):
                load_position_reference("position.json", node, 0.05, 0.05)

    def test_move_spine_can_be_requested_during_read_only_planning(self):
        import task1_precontact_check
        with patch.object(task1_precontact_check, "Ros2MissionNode") as node_type:
            node = node_type.return_value
            node.wait_for_robot_state.return_value = None
            node.sensors.joint_vector.return_value = np.array([0.0])
            with patch.object(task1_precontact_check, "_current_arm_state", return_value=(
                np.zeros(6), np.zeros(6), 0.0, 0.0,
            )):
                with patch.object(task1_precontact_check, "solve_bimanual_pose", return_value={
                    "left_joint_target": [0.0] * 6, "right_joint_target": [0.0] * 6,
                }):
                    with patch.object(task1_precontact_check, "_traverse_pair") as traverse:
                        with patch.object(task1_precontact_check, "_traverse_spine"):
                            with patch.object(task1_precontact_check.Path, "write_text"):
                                with patch.object(task1_precontact_check.Path, "mkdir"):
                                    with patch.object(task1_precontact_check, "locate_pink", return_value={
                                        "center_world": [-1.0, 2.2, 0.834], "center_base": [0.54, 0.0, 0.834],
                                    }):
                                        self.assertEqual(
                                            task1_precontact_check.main(["--stage", "approach", "--move-spine"]), 0
                                        )
                        traverse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
