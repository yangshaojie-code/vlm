import unittest
from types import SimpleNamespace

from ros_mission_executor import RosMissionExecutor


class MissionSettlementTests(unittest.TestCase):
    def make_executor(self, task=3, score=0, phase="-"):
        executor = RosMissionExecutor.__new__(RosMissionExecutor)
        info = SimpleNamespace(task=task, score=score, phase=phase)
        executor.node = SimpleNamespace(
            latest_score=score,
            orchestrator=SimpleNamespace(last_game_info=info),
            spin_once=lambda _timeout: None,
        )
        return executor

    def test_final_task_accepts_referee_score_increase(self):
        executor = self.make_executor(score=25)
        self.assertTrue(executor._wait_server_settlement(3, 0.1, baseline_score=0, baseline_info_score=0))

    def test_task_does_not_settle_without_referee_progress(self):
        executor = self.make_executor(task=2, score=25)
        self.assertFalse(executor._wait_server_settlement(2, 0.0, baseline_score=25, baseline_info_score=25))


if __name__ == "__main__":
    unittest.main()
