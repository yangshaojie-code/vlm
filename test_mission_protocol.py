import unittest

from mission_protocol import MissionProtocolError, parse_gameinfo_payload, parse_mission_payload, parse_score_payload


PAYLOAD = {
    "tasks": [
        {"task": 1, "instruction": "桌面粉色箱到空货架层", "target_color": "pink", "target_body": "pink_box", "place_world": [1, 2, 0.8], "place_type": "shelf_empty", "place_radius": 0.08},
        {"task": 2, "instruction": "货架黄色箱回桌面", "target_color": "黄色", "target_body": "yellow_box", "place_world": {"x": 1, "y": 2, "z": 0.8}, "place_type": "table_side", "place_radius": 0.08},
        {"task": 3, "instruction": "褐色方块放障碍物左边", "target_color": "brown", "target_body": "brown_box", "place_type": "obstacle_left", "place_radius": 0.12},
    ]
}


class MissionProtocolTests(unittest.TestCase):
    def test_parse_formal_mission(self):
        mission = parse_mission_payload(PAYLOAD)
        self.assertEqual([task.task for task in mission.tasks], [1, 2, 3])
        self.assertEqual(mission.task(2).target_color, "yellow")
        self.assertEqual(mission.task(1).place_world, (1.0, 2.0, 0.8))

    def test_parse_string_and_aliases(self):
        mission = parse_mission_payload('{"tasks": ' + __import__("json").dumps(PAYLOAD["tasks"], ensure_ascii=False) + '}')
        self.assertEqual(mission.task(3).target_color, "brown")

    def test_reject_missing_task(self):
        with self.assertRaises(MissionProtocolError):
            parse_mission_payload(PAYLOAD["tasks"][:2])

    def test_single_task_object_is_not_confused_with_an_envelope(self):
        with self.assertRaises(MissionProtocolError):
            parse_mission_payload(PAYLOAD["tasks"][0])

    def test_gameinfo_and_score(self):
        info = parse_gameinfo_payload('{"time": 12.5, "current_task": 2, "attempt": 1, "phase": "running"}')
        self.assertEqual(info.task, 2)
        self.assertAlmostEqual(info.time_seconds, 12.5)
        self.assertEqual(parse_score_payload('{"score": 40}'), 40)

    def test_gameinfo_server_status_line(self):
        info = parse_gameinfo_payload("t=31.3s score=20 task=2/3 best=[40, 0, 0] attempt=1 step=PLACE")
        self.assertAlmostEqual(info.time_seconds, 31.3)
        self.assertEqual(info.task, 2)
        self.assertEqual(info.attempt, 1)
        self.assertEqual(info.score, 20)
        self.assertEqual(info.phase, "PLACE")
        self.assertEqual(info.best_scores, (40.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
