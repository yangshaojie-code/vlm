import unittest

import numpy as np

from task1_transport_check import (
    CARRY_CLEARANCE_MAX_M,
    MIN_SQUEEZE_HOLD_SAMPLES,
    reverse_target,
    transport_command,
)


class Task1TransportCheckTests(unittest.TestCase):
    def test_reverse_target_keeps_yaw_and_moves_away_along_heading(self):
        target = reverse_target([1.0, 2.0, np.pi / 2.0], 0.20)
        np.testing.assert_allclose(target, [1.0, 1.8, np.pi / 2.0], atol=1e-12)

    def test_transport_commands_reverse_then_stops_at_target(self):
        start = np.array([0.0, 0.0, 0.0])
        target = reverse_target(start, 0.20)
        linear, angular, details = transport_command(start, start, target, -1, 0.03, 0.05)
        self.assertLess(linear, 0.0)
        self.assertLessEqual(abs(linear), 0.08)
        self.assertGreaterEqual(abs(linear), 0.04)
        self.assertEqual(details["phase"], "translate")
        almost = target.copy()
        almost[0] += 0.025
        linear, angular, details = transport_command(almost, start, target, -1, 0.03, 0.05)
        self.assertEqual(details["phase"], "complete")
        linear, angular, details = transport_command(target, start, target, -1, 0.03, 0.05)
        self.assertEqual((linear, angular, details["phase"]), (0.0, 0.0, "complete"))

    def test_transport_completes_after_nearly_the_full_requested_distance(self):
        start = np.array([0.0, 0.0, 0.0])
        target = reverse_target(start, 0.20)
        almost = start.copy()
        almost[0] = -0.191
        linear, angular, details = transport_command(almost, start, target, -1, 0.03, 0.05)
        self.assertEqual(details["phase"], "complete")
        self.assertGreaterEqual(details["traveled_m"], 0.19)

    def test_transport_rejects_distance_outside_competition_minimum(self):
        with self.assertRaises(ValueError):
            reverse_target([0.0, 0.0, 0.0], 0.19)

    def test_carry_requires_tighter_than_open_clearance_and_sustained_squeeze(self):
        self.assertLessEqual(CARRY_CLEARANCE_MAX_M, 0.0105)
        self.assertGreater(CARRY_CLEARANCE_MAX_M, 0.0)
        self.assertFalse(0.02 <= CARRY_CLEARANCE_MAX_M)
        self.assertTrue(0.01 <= CARRY_CLEARANCE_MAX_M)
        self.assertGreaterEqual(MIN_SQUEEZE_HOLD_SAMPLES, 16)


if __name__ == "__main__":
    unittest.main()
