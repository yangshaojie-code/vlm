import unittest

import numpy as np

from task1_transport_check import reverse_target, transport_command


class Task1TransportCheckTests(unittest.TestCase):
    def test_reverse_target_keeps_yaw_and_moves_away_along_heading(self):
        target = reverse_target([1.0, 2.0, np.pi / 2.0], 0.20)
        np.testing.assert_allclose(target, [1.0, 1.8, np.pi / 2.0], atol=1e-12)

    def test_transport_commands_reverse_then_stops_at_target(self):
        start = np.array([0.0, 0.0, 0.0])
        target = reverse_target(start, 0.20)
        linear, angular, details = transport_command(start, start, target, -1, 0.02, 0.05)
        self.assertLess(linear, 0.0)
        self.assertEqual(details["phase"], "translate")
        linear, angular, details = transport_command(target, start, target, -1, 0.02, 0.05)
        self.assertEqual((linear, angular, details["phase"]), (0.0, 0.0, "complete"))

    def test_transport_rejects_distance_outside_competition_minimum(self):
        with self.assertRaises(ValueError):
            reverse_target([0.0, 0.0, 0.0], 0.19)


if __name__ == "__main__":
    unittest.main()
