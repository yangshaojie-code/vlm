"""Offline tests for base-frame consistency and feedback-driven state changes."""

import unittest

import numpy as np

from client_common import _rotate_pose_between_base_frames, run_pick_place_loop
from depth_utils import DepthSamplingError, robust_depth_from_bbox
from grasp_pose import place_pose
from motion_planning import IKBackend


class PositionIK(IKBackend):
    def solve(self, target_pos, target_rmat, arm, slide_pos, q_ref):
        target = np.asarray(target_pos, dtype=float)
        return np.array([target[0], target[1], target[2], 0.1, 0.2, 0.3])


def make_poses():
    rotation = np.eye(3)
    pick = {
        "position": np.array([0.8, 0.1, 0.9]),
        "object_position": np.array([0.8, 0.1, 0.8]),
        "rotation": rotation,
    }
    place = {"position": np.array([0.7, -0.2, 0.9]), "rotation": rotation}
    return pick, place


class ControlFlowTests(unittest.TestCase):
    def test_pose_is_rotated_into_final_base_frame(self):
        pose = {
            "position": np.array([1.0, 0.0, 0.5]),
            "object_position": np.array([1.0, 0.0, 0.4]),
            "rotation": np.eye(3),
        }
        result = _rotate_pose_between_base_frames(pose, 0.0, np.pi / 2)
        np.testing.assert_allclose(result["position"], [0.0, -1.0, 0.5], atol=1e-7)
        np.testing.assert_allclose(result["object_position"], [0.0, -1.0, 0.4], atol=1e-7)

    def test_robust_depth_ignores_bbox_edges_and_invalid_values(self):
        depth = np.full((100, 100), 4.0)
        depth[30:70, 30:70] = 0.8
        depth[45:50, 45:50] = np.nan
        depth[50, 50] = 9.0
        value = robust_depth_from_bbox(depth, [20, 20, 80, 80], center_ratio=0.5)
        self.assertAlmostEqual(value, 0.8)

    def test_robust_depth_rejects_missing_measurements(self):
        with self.assertRaises(DepthSamplingError):
            robust_depth_from_bbox(np.zeros((20, 20)), [2, 2, 18, 18])

    def test_task_left_is_rotated_into_current_base_frame(self):
        # Mock the camera projection so only the direction offset is under test.
        import grasp_pose

        original = grasp_pose.pixel_depth_to_base_point
        grasp_pose.pixel_depth_to_base_point = lambda u, v, d: np.zeros(3)
        try:
            pose = place_pose(
                (10, 10),
                1.0,
                "left",
                reference_category="tool_bucket",
                task_to_base_yaw=-np.pi / 2,
            )
        finally:
            grasp_pose.pixel_depth_to_base_point = original
        self.assertGreater(pose["position"][0], 0.0)
        self.assertAlmostEqual(pose["position"][1], 0.0, places=7)

    def test_loop_waits_for_stable_joint_feedback(self):
        pick, place = make_poses()
        qpos = np.zeros(6)
        calls = {}

        def apply_action(action, state):
            calls[state] = calls.get(state, 0) + 1
            target = action[12:18]
            qpos[:] += 0.7 * (target - qpos)

        task = run_pick_place_loop(
            pick,
            place,
            PositionIK(),
            "r",
            lambda: qpos.copy(),
            lambda: 0.1,
            apply_action,
            joint_tolerance=0.01,
            stable_steps=3,
            gripper_hold_steps=4,
            max_steps_per_state=100,
        )

        self.assertTrue(task.is_done())
        self.assertEqual(calls["CLOSE_GRIPPER"], 4)
        self.assertEqual(calls["OPEN_GRIPPER"], 4)
        self.assertGreater(calls["APPROACH_PICK"], 3)

    def test_loop_times_out_without_feedback(self):
        pick, place = make_poses()
        with self.assertRaisesRegex(RuntimeError, "APPROACH_PICK"):
            run_pick_place_loop(
                pick,
                place,
                PositionIK(),
                "r",
                lambda: np.zeros(6),
                lambda: 0.1,
                lambda action, state: None,
                stable_steps=2,
                max_steps_per_state=3,
            )


if __name__ == "__main__":
    unittest.main()
