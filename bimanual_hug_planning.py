"""Pure planning helpers for the validated Task 1 bimanual hug posture.

This module intentionally has no ROS dependencies.  It captures the official
fixed-layout high pre-grasp geometry used by the no-contact and inward checks,
so later execution code can share one bounded, FK-verified plan.
"""

from __future__ import annotations

import math

import numpy as np

from motion_planning import MMK2KdlBackend


PRE_GRASP_FWD = 0.48240646
PRE_GRASP_LAT = 0.22475936
PRE_GRASP_Z0 = 1.32163718
MAX_INWARD_DELTA = 0.04
FK_POSITION_TOLERANCE_M = 1e-5
# Competition carton half-extents: 0.24 x 0.16 x 0.19 m.
COMPETITION_BOX_HALF_M = (0.12, 0.08, 0.095)
# Official-client leftover: +0.065 m past a reconstructed center rides the
# far lip of the 0.16 m table/cube depth and is not a max-contact hug.
LEGACY_FAR_LIP_GRASP_FWD_M = 0.065

LEFT_A_ROT = np.array([
    [0.99890619, 0.04294831, 0.01848963],
    [-0.02030260, 0.04216758, 0.99890425],
    [0.04212158, -0.99818703, 0.04299342],
])
RIGHT_A_ROT = np.array([
    [0.99890619, -0.04294831, 0.01848963],
    [0.02030260, 0.04216758, -0.99890425],
    [0.04212158, 0.99818703, 0.04299342],
])


def side_face_hug_geometry(
    half_extents=COMPETITION_BOX_HALF_M,
    squeeze_axis: str = "x",
    box_point: str = "center",
) -> dict:
    """Palm targets that cover the center of the squeezed side faces.

    ``squeeze_axis`` is the box-frame axis the palms close along (``x`` for
    the 0.24 m table/cube faces, ``y`` for the 0.16 m shelf-front faces).
    Callers that already reconstructed the geometric center must pass
    ``box_point="center"`` so no extra forward offset is added.  A raw
    front-face RGB-D hit uses ``box_point="front_face"``, which shifts
    forward by the remaining half-depth to reach that same center.
    """
    extents = np.asarray(half_extents, dtype=float)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError("half_extents must be a finite 3-vector")
    hx, hy, hz = (float(value) for value in extents)
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("half_extents must be positive")
    if squeeze_axis == "x":
        squeeze_half, depth_half = hx, hy
        hold_half = min(squeeze_half, 0.115)
        approach_cap = 0.13
    elif squeeze_axis == "y":
        squeeze_half, depth_half = hy, hx
        hold_half = squeeze_half
        approach_cap = 0.10
    else:
        raise ValueError("squeeze_axis must be 'x' or 'y'")
    if box_point == "center":
        grasp_fwd = 0.0
    elif box_point == "front_face":
        grasp_fwd = depth_half
    else:
        raise ValueError("box_point must be 'center' or 'front_face'")
    return {
        "hold_half_m": float(hold_half),
        "approach_half_m": float(min(approach_cap, hold_half + 0.02)),
        "grasp_fwd_offset_m": float(grasp_fwd),
        "grasp_z_offset_m": float(min(0.02, 0.30 * hz)),
        "squeeze_half_m": float(squeeze_half),
        "depth_half_m": float(depth_half),
    }


def lateral_for_inward(inward_delta: float) -> float:
    """Convert a symmetric inward distance to the remaining lateral offset."""
    inward_delta = float(inward_delta)
    if not np.isfinite(inward_delta) or not 0.0 <= inward_delta <= MAX_INWARD_DELTA:
        raise ValueError(f"inward delta must be within [0.0, {MAX_INWARD_DELTA}] m")
    return PRE_GRASP_LAT - inward_delta


def pregrasp_targets(slide: float, lateral: float = PRE_GRASP_LAT) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric official high-posture end-effector positions."""
    slide = float(slide)
    lateral = float(lateral)
    if not np.isfinite(slide):
        raise ValueError("slide must be finite")
    minimum = lateral_for_inward(MAX_INWARD_DELTA)
    if not minimum <= lateral <= PRE_GRASP_LAT:
        raise ValueError(f"pre-grasp lateral offset must be within [{minimum}, {PRE_GRASP_LAT}]")
    z = PRE_GRASP_Z0 - slide
    return (
        np.array([PRE_GRASP_FWD, lateral, z]),
        np.array([PRE_GRASP_FWD, -lateral, z]),
    )


def solve_bimanual_hug_pose(
    slide: float,
    left_current,
    right_current,
    lateral: float = PRE_GRASP_LAT,
    backend=None,
) -> dict:
    """Solve and FK-verify one paired high hug pose without ROS I/O."""
    left_target, right_target = pregrasp_targets(slide, lateral)
    result = solve_bimanual_pose(
        slide,
        left_current,
        right_current,
        left_target,
        right_target,
        backend=backend,
    )
    result["lateral_offset_m"] = float(lateral)
    return result


def solve_bimanual_pose(
    slide: float,
    left_current,
    right_current,
    left_target,
    right_target,
    backend=None,
) -> dict:
    """Solve and FK-verify a paired pose at arbitrary Cartesian positions."""
    left_current = np.asarray(left_current, dtype=float)
    right_current = np.asarray(right_current, dtype=float)
    if left_current.shape != (6,) or right_current.shape != (6,):
        raise ValueError("both arm references must contain six joints")
    if not np.all(np.isfinite(left_current)) or not np.all(np.isfinite(right_current)):
        raise ValueError("both arm references must be finite")
    backend = backend or MMK2KdlBackend()
    left_target = np.asarray(left_target, dtype=float)
    right_target = np.asarray(right_target, dtype=float)
    if left_target.shape != (3,) or right_target.shape != (3,):
        raise ValueError("both Cartesian targets must contain three values")
    if not np.all(np.isfinite(left_target)) or not np.all(np.isfinite(right_target)):
        raise ValueError("both Cartesian targets must be finite")
    solved_slide, left_solution, right_solution = backend.solve_bimanual(
        left_target,
        LEFT_A_ROT,
        right_target,
        RIGHT_A_ROT,
        left_current,
        right_current,
        slide_reference=float(slide),
        slide_pos=float(slide),
    )
    left_fk = backend.forward("l", solved_slide, left_solution)
    right_fk = backend.forward("r", solved_slide, right_solution)
    left_error = float(np.linalg.norm(left_fk[:3, 3] - left_target))
    right_error = float(np.linalg.norm(right_fk[:3, 3] - right_target))
    if left_error > FK_POSITION_TOLERANCE_M or right_error > FK_POSITION_TOLERANCE_M:
        raise ValueError(f"KDL FK verification failed: left={left_error}, right={right_error}")
    return {
        "slide": float(solved_slide),
        "left_target_position": left_target.tolist(),
        "right_target_position": right_target.tolist(),
        "left_joint_target": np.asarray(left_solution).tolist(),
        "right_joint_target": np.asarray(right_solution).tolist(),
        "left_fk_position": left_fk[:3, 3].tolist(),
        "right_fk_position": right_fk[:3, 3].tolist(),
        "left_fk_error_m": left_error,
        "right_fk_error_m": right_error,
    }


def build_bimanual_hug_plan(
    slide: float,
    left_current,
    right_current,
    inward_delta: float = 0.0,
    max_step: float = 0.10,
    backend=None,
) -> dict:
    """Build a bounded pre-grasp/inward/return plan for paired arm control."""
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be positive and finite")
    pregrasp = solve_bimanual_hug_pose(slide, left_current, right_current, backend=backend)
    inward = None
    if float(inward_delta):
        inward = solve_bimanual_hug_pose(
            slide,
            pregrasp["left_joint_target"],
            pregrasp["right_joint_target"],
            lateral=lateral_for_inward(inward_delta),
            backend=backend,
        )
    left_initial = np.asarray(left_current, dtype=float)
    right_initial = np.asarray(right_current, dtype=float)
    pregrasp_steps = waypoint_count(
        left_initial,
        pregrasp["left_joint_target"],
        right_initial,
        pregrasp["right_joint_target"],
        max_step,
    )
    inward_steps = 0 if inward is None else waypoint_count(
        pregrasp["left_joint_target"],
        inward["left_joint_target"],
        pregrasp["right_joint_target"],
        inward["right_joint_target"],
        max_step,
    )
    return {
        "pregrasp": pregrasp,
        "inward": inward,
        "pregrasp_waypoint_count": pregrasp_steps,
        "inward_waypoint_count": inward_steps,
        "max_step_rad": max_step,
        "inward_delta_m": float(inward_delta),
    }


def waypoint_count(left_initial, left_target, right_initial, right_target, max_step: float) -> int:
    """Return the number of paired waypoints needed to bound each joint step."""
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be positive and finite")
    left_initial = np.asarray(left_initial, dtype=float)
    left_target = np.asarray(left_target, dtype=float)
    right_initial = np.asarray(right_initial, dtype=float)
    right_target = np.asarray(right_target, dtype=float)
    if any(value.shape != (6,) for value in (left_initial, left_target, right_initial, right_target)):
        raise ValueError("all arm joint vectors must contain six values")
    delta = max(
        float(np.max(np.abs(left_target - left_initial))),
        float(np.max(np.abs(right_target - right_initial))),
    )
    return max(1, int(math.ceil(delta / max_step)))
