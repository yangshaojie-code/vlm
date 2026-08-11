"""Task-specific placement context without hard-coded scene coordinates."""

from typing import Any, Dict, Optional, Tuple

from mission_protocol import TaskSpec


def remember_task1_source(context: Dict[str, Any], world_point, table_side: Optional[str] = None) -> None:
    """Persist task-one's original table slot for task two."""
    point = tuple(float(value) for value in world_point)
    if len(point) != 3:
        raise ValueError("task1 source point 必须是三维坐标")
    context["task1_source_world"] = point
    if table_side is not None:
        context["task1_table_side"] = str(table_side)


def resolve_place_world(task: TaskSpec, context: Dict[str, Any]) -> Tuple[float, float, float]:
    """Resolve a task's placement point from Server data and mission context.

    Task two may use the source point captured before task one moved its box.
    Server-provided ``place_world`` remains authoritative whenever present.
    """
    if task.place_world is not None:
        return task.place_world
    if task.task == 2 and context.get("task1_source_world") is not None:
        return tuple(context["task1_source_world"])
    raise ValueError(f"任务 {task.task} 缺少 place_world，且没有可用的任务上下文")


def validate_task_context(task: TaskSpec, context: Dict[str, Any]) -> None:
    """Fail before motion starts when a task cannot be planned safely."""
    resolve_place_world(task, context)
    if task.place_radius is None:
        raise ValueError(f"任务 {task.task} 缺少 place_radius")

