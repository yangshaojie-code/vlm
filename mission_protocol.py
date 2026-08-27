"""Protocol models for the formal three-task material-sorting mission.

The official Server publishes JSON through ROS 2 ``std_msgs/msg/String``.
This module deliberately has no ROS dependency so the message contract and
mission logic can be tested before the official container is available.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class MissionProtocolError(ValueError):
    """Raised when a Server message does not match the formal contract."""


COLOR_ALIASES = {
    "pink": "pink",
    "pink_box": "pink",
    "粉色": "pink",
    "yellow": "yellow",
    "yellow_box": "yellow",
    "黄色": "yellow",
    "brown": "brown",
    "brown_box": "brown",
    "棕色": "brown",
    "褐色": "brown",
    "褐色方块": "brown",
    "white": "white",
    "white_box": "white",
    "packaging_box": "white",
    "白色": "white",
    "白色长方体": "white",
}


def _decode_payload(payload: Any) -> Any:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        value = payload.strip()
        if not value:
            raise MissionProtocolError("收到空的 Server 消息")
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise MissionProtocolError(f"消息不是合法 JSON: {value[:160]!r}") from exc
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MissionProtocolError(f"{name} 必须是 JSON 对象，收到 {type(value).__name__}")
    return value


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_color(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    color = COLOR_ALIASES.get(key)
    if color is None:
        raise MissionProtocolError(f"不支持的目标颜色: {value!r}")
    return color


def _vec3(value: Any, name: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = [_first(value, "x"), _first(value, "y"), _first(value, "z")]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MissionProtocolError(f"{name} 不是合法坐标") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise MissionProtocolError(f"{name} 必须是长度至少为 3 的坐标")
    try:
        return tuple(float(value[i]) for i in range(3))
    except (TypeError, ValueError) as exc:
        raise MissionProtocolError(f"{name} 包含非数字坐标") from exc


@dataclass(frozen=True)
class TaskSpec:
    """One formal task published in ``/material/instruction``."""

    task: int
    instruction: str
    target_color: str
    target_body: Optional[str] = None
    place_world: Optional[Tuple[float, float, float]] = None
    place_type: Optional[str] = None
    place_radius: Optional[float] = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = _mapping(value, "任务")
        task_value = _first(data, "task", "task_id", "index")
        try:
            task = int(task_value)
        except (TypeError, ValueError) as exc:
            raise MissionProtocolError(f"任务编号无效: {task_value!r}") from exc
        if task not in (1, 2, 3):
            raise MissionProtocolError(f"正式任务编号必须是 1/2/3，收到 {task}")

        color = normalize_color(_first(data, "target_color", "color"))
        if color is None:
            raise MissionProtocolError(f"任务 {task} 缺少 target_color")
        instruction = str(_first(data, "instruction", "text", default="")).strip()
        if not instruction:
            instruction = f"task {task}: {color}"
        target_body = _first(data, "target_body", "target")
        if target_body is not None:
            target_body = str(target_body)
        radius = _first(data, "place_radius", "radius")
        if radius is not None:
            try:
                radius = float(radius)
            except (TypeError, ValueError) as exc:
                raise MissionProtocolError(f"任务 {task} 的 place_radius 无效") from exc
            if radius <= 0:
                raise MissionProtocolError(f"任务 {task} 的 place_radius 必须为正数")

        return cls(
            task=task,
            instruction=instruction,
            target_color=color,
            target_body=target_body,
            place_world=_vec3(_first(data, "place_world", "place_position"), "place_world"),
            place_type=_first(data, "place_type", "placement_type"),
            place_radius=radius,
            raw=dict(data),
        )


@dataclass(frozen=True)
class MissionSpec:
    tasks: Tuple[TaskSpec, ...]
    raw: Any = field(default=None, repr=False)

    def task(self, task_id: int) -> TaskSpec:
        for task in self.tasks:
            if task.task == task_id:
                return task
        raise MissionProtocolError(f"未找到任务 {task_id}")


def parse_mission_payload(payload: Any) -> MissionSpec:
    """Parse the JSON list published on ``/material/instruction``."""
    value = _decode_payload(payload)
    if isinstance(value, Mapping):
        # A formal envelope uses ``tasks``/``mission``. A single task object
        # also contains an ``instruction`` field, so do not mistake that text
        # for an envelope when ``task`` is present.
        if "tasks" in value or "mission" in value:
            value = _first(value, "tasks", "mission")
            value = _decode_payload(value)
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise MissionProtocolError("/material/instruction 必须是任务 JSON 列表")

    tasks = tuple(TaskSpec.from_mapping(item) for item in value)
    if tuple(task.task for task in tasks) != (1, 2, 3):
        raise MissionProtocolError(
            "正式任务必须按顺序包含 task=1、task=2、task=3，"
            f"收到 {[task.task for task in tasks]}"
        )
    return MissionSpec(tasks=tasks, raw=value)


@dataclass(frozen=True)
class GameInfo:
    """Best-effort normalized view of ``/referee/gameinfo``."""

    time_seconds: Optional[float] = None
    task: Optional[int] = None
    attempt: Optional[int] = None
    phase: Optional[str] = None
    score: Optional[float] = None
    best_scores: Tuple[float, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


def parse_gameinfo_payload(payload: Any) -> GameInfo:
    raw_payload = getattr(payload, "data", payload)
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str) and not raw_payload.lstrip().startswith(("{", "[")):
        # The current Server publishes a compact status line rather than JSON:
        # t=31.3s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-
        pairs = dict(re.findall(r"(?<![A-Za-z0-9_])(t|score|task|attempt|step)=([^\s]+)", raw_payload))
        if not pairs:
            raise MissionProtocolError(f"gameinfo is neither JSON nor key/value text: {raw_payload[:160]!r}")
        value = {
            "time": pairs.get("t", "").removesuffix("s") or None,
            "score": pairs.get("score"),
            "task": pairs.get("task", "").split("/", 1)[0] or None,
            "attempt": pairs.get("attempt"),
            "phase": pairs.get("step"),
            "status_text": raw_payload,
        }
        best_match = re.search(r"(?<![A-Za-z0-9_])best=\[([^\]]*)\]", raw_payload)
        if best_match:
            try:
                value["best_scores"] = tuple(float(item.strip()) for item in best_match.group(1).split(",") if item.strip())
            except ValueError as exc:
                raise MissionProtocolError(f"invalid gameinfo best scores: {best_match.group(1)!r}") from exc
    else:
        value = _mapping(_decode_payload(raw_payload), "gameinfo")
    time_value = _first(value, "time", "sim_time", "elapsed", "elapsed_time", "time_seconds")
    task_value = _first(value, "task", "task_id", "current_task")
    attempt_value = _first(value, "attempt", "attempts", "try", "try_count")
    score_value = _first(value, "score", "total_score")
    try:
        time_seconds = None if time_value is None else float(time_value)
        task = None if task_value is None else int(task_value)
        attempt = None if attempt_value is None else int(attempt_value)
        score = None if score_value is None else float(score_value)
    except (TypeError, ValueError) as exc:
        raise MissionProtocolError("gameinfo 含有无法解析的数值字段") from exc
    best_value = _first(value, "best_scores", "best", default=()) or ()
    if isinstance(best_value, str):
        try:
            best_value = json.loads(best_value)
        except json.JSONDecodeError:
            best_value = [item for item in best_value.strip("[]").split(",") if item.strip()]
    if not isinstance(best_value, Sequence) or isinstance(best_value, (bytes, bytearray, str)):
        best_value = (best_value,)
    try:
        best_scores = tuple(float(item) for item in best_value)
    except (TypeError, ValueError) as exc:
        raise MissionProtocolError("gameinfo best scores contain invalid values") from exc
    return GameInfo(
        time_seconds=time_seconds,
        task=task,
        attempt=attempt,
        phase=_first(value, "phase", "state", "status"),
        score=score,
        best_scores=best_scores,
        raw=dict(value),
    )


def parse_score_payload(payload: Any) -> Optional[int]:
    value = _decode_payload(payload)
    if isinstance(value, Mapping):
        value = _first(value, "score", "total_score")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MissionProtocolError(f"score 消息无效: {value!r}") from exc
