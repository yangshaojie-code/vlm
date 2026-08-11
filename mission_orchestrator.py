"""Environment-independent state machine for one formal three-task mission."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mission_protocol import GameInfo, MissionSpec, TaskSpec, parse_gameinfo_payload, parse_mission_payload


class MissionState(str, Enum):
    GAME_INIT = "GAME_INIT"
    TASK_ATTEMPT = "TASK_ATTEMPT"
    TASK_SETTLE = "TASK_SETTLE"
    GAME_DONE = "GAME_DONE"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class AttemptRecord:
    task: int
    attempt: int
    success: bool
    score: Optional[float] = None
    reason: Optional[str] = None


class MissionStateError(RuntimeError):
    """Raised when the client advances the mission in an invalid order."""


class MissionOrchestrator:
    """Track task order without resetting physical state or randomization.

    ROS callbacks and motion code call this class; it intentionally does not
    import rclpy and is therefore usable in unit tests and offline simulation.
    """

    def __init__(self, max_attempts: int = 3, time_limit_seconds: float = 600.0):
        if max_attempts < 1 or time_limit_seconds <= 0:
            raise ValueError("max_attempts 和 time_limit_seconds 必须为正数")
        self.max_attempts = int(max_attempts)
        self.time_limit_seconds = float(time_limit_seconds)
        self.mission: Optional[MissionSpec] = None
        self.state = MissionState.GAME_INIT
        self.task_index = 0
        self.attempts: Dict[int, int] = {}
        self.best_scores: Dict[int, float] = {}
        self.records: List[AttemptRecord] = []
        self.context: Dict[str, Any] = {}
        self.elapsed_seconds = 0.0
        self.last_game_info: Optional[GameInfo] = None

    @property
    def current_task(self) -> Optional[TaskSpec]:
        if self.mission is None or self.task_index >= len(self.mission.tasks):
            return None
        return self.mission.tasks[self.task_index]

    @property
    def current_attempt(self) -> int:
        task = self.current_task
        return 0 if task is None else self.attempts.get(task.task, 0)

    def load_mission(self, payload: Any) -> MissionSpec:
        if self.state not in (MissionState.GAME_INIT, MissionState.GAME_DONE, MissionState.TIMEOUT):
            raise MissionStateError("不能在任务执行中替换 mission")
        self.mission = payload if isinstance(payload, MissionSpec) else parse_mission_payload(payload)
        self.state = MissionState.GAME_INIT
        self.task_index = 0
        self.attempts.clear()
        self.best_scores.clear()
        self.records.clear()
        self.context.clear()
        self.elapsed_seconds = 0.0
        self.last_game_info = None
        return self.mission

    def start_game(self) -> TaskSpec:
        if self.mission is None:
            raise MissionStateError("尚未加载 /material/instruction")
        if self.state != MissionState.GAME_INIT:
            raise MissionStateError(f"当前状态 {self.state} 不能开始游戏")
        self.state = MissionState.TASK_ATTEMPT
        return self.current_task

    def start_attempt(self) -> TaskSpec:
        self._check_deadline()
        if self.state != MissionState.TASK_ATTEMPT or self.current_task is None:
            raise MissionStateError(f"当前状态 {self.state} 不能开始尝试")
        task_id = self.current_task.task
        used = self.attempts.get(task_id, 0)
        if used >= self.max_attempts:
            raise MissionStateError(f"任务 {task_id} 已用尽 {self.max_attempts} 次机会")
        self.attempts[task_id] = used + 1
        return self.current_task

    def complete_attempt(self, success: bool, score: Optional[float] = None, reason: Optional[str] = None) -> AttemptRecord:
        if self.state != MissionState.TASK_ATTEMPT or self.current_task is None:
            raise MissionStateError(f"当前状态 {self.state} 不能结算尝试")
        task_id = self.current_task.task
        record = AttemptRecord(task_id, self.current_attempt, bool(success), score, reason)
        self.records.append(record)
        if score is not None:
            self.best_scores[task_id] = max(score, self.best_scores.get(task_id, float("-inf")))
        self.state = MissionState.TASK_SETTLE
        return record

    def settle(self) -> Optional[TaskSpec]:
        if self.state != MissionState.TASK_SETTLE or self.current_task is None:
            raise MissionStateError(f"当前状态 {self.state} 不能完成结算")
        task_id = self.current_task.task
        latest = self.records[-1]
        if latest.task != task_id:
            raise MissionStateError("尝试记录与当前任务不一致")
        if latest.success or self.current_attempt >= self.max_attempts:
            self.task_index += 1
        if self.task_index >= 3:
            self.state = MissionState.GAME_DONE
            return None
        self.state = MissionState.TASK_ATTEMPT
        return self.current_task

    def record_recovery(self, action: str) -> None:
        if self.state != MissionState.TASK_ATTEMPT:
            raise MissionStateError("只有任务尝试期间允许记录局部恢复")
        self.context.setdefault("recoveries", []).append({"task": self.current_task.task, "attempt": self.current_attempt, "action": str(action)})

    def set_context(self, key: str, value: Any) -> None:
        self.context[str(key)] = value

    def sync_game_info(self, payload: Any) -> GameInfo:
        info = payload if isinstance(payload, GameInfo) else parse_gameinfo_payload(payload)
        self.last_game_info = info
        if info.time_seconds is not None:
            self.elapsed_seconds = max(self.elapsed_seconds, info.time_seconds)
        if self.elapsed_seconds >= self.time_limit_seconds and self.state not in (MissionState.GAME_DONE, MissionState.TIMEOUT):
            self.state = MissionState.TIMEOUT
        return info

    def _check_deadline(self) -> None:
        if self.elapsed_seconds >= self.time_limit_seconds:
            self.state = MissionState.TIMEOUT
            raise MissionStateError("已达到 600 秒仿真时间上限")
