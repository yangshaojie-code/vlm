import unittest

from formal_mission_runtime import AttemptResult, MissionRuntime
from mission_orchestrator import MissionState
from test_mission_protocol import PAYLOAD


class FormalMissionRuntimeTests(unittest.TestCase):
    def test_runs_three_tasks_and_preserves_context(self):
        calls = []

        def execute(task, context, attempt):
            calls.append((task.task, attempt))
            context.setdefault("seen", []).append(task.task)
            return AttemptResult(True, score=task.task)

        runtime = MissionRuntime(execute)
        engine = runtime.run(PAYLOAD)
        self.assertEqual(engine.state, MissionState.GAME_DONE)
        self.assertEqual(calls, [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(engine.context["seen"], [1, 2, 3])

    def test_executor_failure_is_retried_without_reset(self):
        calls = []

        def execute(task, context, attempt):
            calls.append((task.task, attempt))
            return AttemptResult(task.task == 1 or attempt == 3)

        runtime = MissionRuntime(execute)
        engine = runtime.run(PAYLOAD)
        self.assertEqual(engine.state, MissionState.GAME_DONE)
        self.assertEqual(calls[:4], [(1, 1), (2, 1), (2, 2), (2, 3)])
        self.assertEqual(engine.attempts[2], 3)


if __name__ == "__main__":
    unittest.main()

