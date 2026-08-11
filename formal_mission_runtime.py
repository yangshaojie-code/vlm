"""Callback-based mission runtime shared by ROS and offline integration tests."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from mission_orchestrator import MissionOrchestrator, MissionState
from mission_protocol import TaskSpec


@dataclass(frozen=True)
class AttemptResult:
    success: bool
    score: Optional[float] = None
    reason: Optional[str] = None


Executor = Callable[[TaskSpec, Dict[str, Any], int], AttemptResult]


class MissionRuntime:
    """Drive the formal mission while leaving physical actions injectable.

    ``execute_attempt`` must perform navigation, perception, manipulation and
    return-to-end-zone for exactly one attempt. It must not reset the scene.
    """

    def __init__(self, execute_attempt: Executor, orchestrator: Optional[MissionOrchestrator] = None):
        self.execute_attempt = execute_attempt
        self.orchestrator = orchestrator or MissionOrchestrator()

    def run(self, instruction_payload: Any) -> MissionOrchestrator:
        self.orchestrator.load_mission(instruction_payload)
        self.orchestrator.start_game()
        while self.orchestrator.state not in (MissionState.GAME_DONE, MissionState.TIMEOUT):
            task = self.orchestrator.start_attempt()
            try:
                result = self.execute_attempt(task, self.orchestrator.context, self.orchestrator.current_attempt)
                if not isinstance(result, AttemptResult):
                    result = AttemptResult(bool(result))
            except Exception as exc:  # keep the client alive for an in-place retry
                result = AttemptResult(False, reason=f"attempt exception: {exc}")
            self.orchestrator.complete_attempt(result.success, result.score, result.reason)
            self.orchestrator.settle()
        return self.orchestrator

