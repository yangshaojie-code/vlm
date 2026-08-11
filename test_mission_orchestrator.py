import unittest

from mission_orchestrator import MissionOrchestrator, MissionState, MissionStateError
from test_mission_protocol import PAYLOAD


class MissionOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.engine = MissionOrchestrator()
        self.engine.load_mission(PAYLOAD)
        self.engine.start_game()

    def test_success_advances_to_next_task_without_reset(self):
        task = self.engine.start_attempt()
        self.assertEqual(task.task, 1)
        self.engine.set_context("task1_table_slot", "left")
        self.engine.complete_attempt(True, score=40)
        self.assertEqual(self.engine.settle().task, 2)
        self.assertEqual(self.engine.context["task1_table_slot"], "left")

    def test_failed_attempt_can_retry_then_exhausts(self):
        for attempt in range(1, 4):
            self.engine.start_attempt()
            self.engine.record_recovery("reobserve")
            self.engine.complete_attempt(False, reason="drop")
            next_task = self.engine.settle()
            if attempt < 3:
                self.assertEqual(next_task.task, 1)
            else:
                self.assertEqual(next_task.task, 2)
        self.assertEqual(self.engine.attempts[1], 3)

    def test_deadline_stops_new_attempt(self):
        self.engine.sync_game_info('{"time": 600}')
        self.assertEqual(self.engine.state, MissionState.TIMEOUT)
        with self.assertRaises(MissionStateError):
            self.engine.start_attempt()

    def test_already_parsed_mission_can_be_loaded(self):
        mission = self.engine.mission
        fresh = MissionOrchestrator()
        fresh.load_mission(mission)
        self.assertEqual(fresh.current_task.task, 1)


if __name__ == "__main__":
    unittest.main()
