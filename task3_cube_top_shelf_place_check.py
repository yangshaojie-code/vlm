"""Task 3 bounded yellow-box cube-top pick and shelf L1 place check.

Grabs the yellow box from the top of the white cube with the verified
bimanual hug geometry at the raised cube-top height, lifts 0.10 m clear of
the cube, backs off the table, and places it on shelf L1 left of the white
packaging-box obstacle.  It never enables the formal executor or returns to
the end zone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from bimanual_hug_planning import PRE_GRASP_Z0, solve_bimanual_hug_pose, solve_bimanual_pose
from color_box_detector import detect_colored_boxes
from depth_utils import robust_depth_from_bbox
from geometry_utils import transform_point
from head_camera_kinematics import SLIDE_LIMITS
from motion_planning import MMK2KdlBackend
from ros2_mission_node import Ros2MissionNode
from task1_bimanual_approach_campaign import (
    _current_arm_state_unbounded,
    _traverse_grippers,
    command_gripper_value,
)
from task1_pick_lift_check import (
    APPROACH_JOINT_TOLERANCE_RAD,
    RETRACTION_JOINT_TOLERANCE_RAD,
    TASK1_APPROACH_HALF_M,
    TASK1_HOLD_HALF_M,
    _approach_until_reached_or_contact,
    blocked_table_hug_lock,
    carry_hold_ok,
    contact_approach_geometry,
    contact_clearance_schedule,
    hug_moved_from_pregrasp,
    lift_slide_target,
)
from task1_precontact_check import (
    BOX_HALF_DEPTH,
    MAX_ARM_STEP,
    MAX_OBSERVE_BACKUP_ATTEMPTS,
    MAX_SPINE_STEP,
    OBSERVE_BACKUP_M,
    OBSERVE_BACKUP_TIMEOUT_SEC,
    _backup_reverse,
    _navigate,
    _traverse_pair,
    _traverse_spine,
    _world_to_base,
    should_backup_to_observe,
    station_target,
    wrap_to_pi,
)
from task1_shelf_place_check import (
    DEFAULT_LINE_TIMEOUT_SEC,
    DEFAULT_NAV_TIMEOUT_SEC,
    DEFAULT_SHELF_TIMEOUT_SEC,
    DEFAULT_YAW_TIMEOUT_SEC,
    MAX_STAGING_NAV_M,
    SHELF_RETREAT_M,
    SHELF_ZONE_X_MAX,
    STAGING_BACK_M,
    TABLE_LEAVE_DISTANCE_M,
    TABLE_LEAVE_MIN_TRAVELED_M,
    _face_yaw_holding,
    _navigate_holding,
    _retract_arms,
    _sleep_holding,
    apply_slide_keep_hold,
    held_center_world,
    pose_offset,
    release_cartesian,
    slide_for_held_z,
    staging_pose,
)
from task1_transport_check import (
    CARRY_CLEARANCE_MAX_M,
    DEFAULT_SQUEEZE_SECONDS,
    _command_hug,
    _confirm_inward_squeeze,
    _odom_pose,
    _traverse_spine_holding,
)
from task2_shelf_to_table_check import (
    _bind_hold_keeper,
    _traverse_spine_keeping_pose,
    already_carrying_box,
    center_from_shelf_front,
    local_carry_hold,
)


TASK_COLOR = "yellow"
YELLOW_FIXED_WORLD = np.array([-0.54, 2.30, 1.004], dtype=float)
CUBE_TOP_Z = 0.909
BOX_HALF_HEIGHT_M = 0.095
CUBE_TOP_CENTER_Z = CUBE_TOP_Z + BOX_HALF_HEIGHT_M
INSTRUCTION_PLACE_WORLD = np.array([-2.68, 0.54, 0.498], dtype=float)
PLACE_RADIUS_M = 0.24
PLACE_ACCEPT_RADIUS_M = 0.08
PLACE_YAW = math.pi
GRASP_YAW = math.pi / 2.0
TASK3_LIFT_HEIGHT_M = 0.10
TASK3_PLACE_CLEARANCE_M = 0.055
# KDL origin is the fingertip / palm front.  grasp_fwd=0 parked that tip on
# the box center, so the pad sat behind the carton.  +0.06 m puts the pad
# mid-face; the far face is +0.08 m, so fingertips stay on the side, not air.
TASK3_GRASP_FWD_OFFSET_M = 0.06
TASK3_GRASP_Z_OFFSET_M = 0.0
BOX_HALF_DEPTH_M = 0.08
BOX_HALF_WIDTH_M = 0.12
PACKAGING_WORLD = np.array([-2.63, 0.778, 0.530], dtype=float)
PACKAGING_HALF_HEIGHT_M = 0.117
PACKAGING_TOP_Z_M = float(PACKAGING_WORLD[2] + PACKAGING_HALF_HEIGHT_M)
SHELF_L1_BOARD_Z = 0.403
SHELF_L2_BOARD_Z = 0.732
# A 0.19 m carton cannot fly over the packaging (top ~0.65 m) and under L2
# (0.732 m).  Cruise inside the L1 opening, south of the packaging, then west.
L2_CLEARANCE_M = 0.08
L1_BOARD_CLEARANCE_M = 0.04
TASK3_APPROACH_Z_M = float(SHELF_L2_BOARD_Z - L2_CLEARANCE_M - BOX_HALF_HEIGHT_M)
# Last pass used 0.02 m/side. Palms ended 0.218 m apart on a 0.24 m box, so
# retreat dragged it out. 0.04 m/side clears the faces; a small -X withdraw
# also slides the fingertips off the wrap before the base backs out.
TASK3_RELEASE_SPREAD_M = 0.04
TASK3_RELEASE_WITHDRAW_M = 0.04
TASK3_RELEASE_OPEN_RAD = 0.10
TASK3_MAX_PLACE_OUTWARD_M = 0.06
TASK3_PLACE_REMAINING_OK_M = 0.06
TASK3_STANDOFF_M = 0.50
TASK3_HOLD_HALF_M = 0.115
TASK3_HOLD_SQUEEZE_RAD = 0.05
TASK3_HOLD_LINEAR_SPEED = 0.42
TASK3_SHELF_LINEAR_SPEED = 0.22
TASK3_TABLE_LEAVE_LINEAR_SPEED = 0.36
TASK3_HOLD_ANGULAR_SPEED = 1.10
TASK3_LINE_ANGULAR_SPEED = 0.95
SHELF_AISLE_Y_M = 0.778
APPROACH_STALL_X_M = -1.83
PLACE_ALIGN_YAW_TOLERANCE_RAD = 0.05
# 0.05 left square-west ~2.7° south of west; that drives the south arm into the post.
PLACE_INSERT_YAW_TOLERANCE_RAD = 0.02
SOUTH_SHIFT_MIN_M = 0.08
MAX_HOLD_REFRESH = 1
# A confirmed squeeze often sits inside 0.115 m (palms on the 0.24 m faces).
# 0.100 was aborting a working hold; 0.122 still catches an open corner graze.
MIN_HOLD_HALF_SPAN_M = 0.075
MAX_HOLD_HALF_SPAN_M = 0.122
MAX_HOLD_PALM_DZ_M = 0.008
MAX_HUG_Y_PULL_M = 0.02
SHELF_BOARD_SURFACES_Z = (0.403, 0.732, 1.061, 1.366, 1.695, 2.024)
# Packaging euler Rx=90: world-Y half is the local Z half 0.051 m, not 0.10 m.
OBSTACLE_HALF_Y_M = 0.051
OBSTACLE_HALF_X_M = 0.09
PLACE_LEFT_GAP_M = 0.06
PLACE_SHELF_INSET_M = 0.05
# MJCF shelf posts at local y=±0.388, half-size 0.02 m.
SHELF_POST_LOCAL_Y_M = 0.388
SHELF_POST_HALF_M = 0.02
# The head camera sits ~0.35 m above the yellow box at the observe distance,
# so a small extra pitch (the body already tilts -0.33) keeps it centered.
OBSERVE_HEAD = (0.0, -0.10)
# From staging the camera is still high; -0.40 looks at L2, not the L1 box.
SHELF_OBSERVE_HEAD = (0.0, -0.58)
TABLE_SOUTH_EDGE_Y = 1.915
STATION_Y_MAX = TABLE_SOUTH_EDGE_Y - 0.03
HUG_WINDOW_X = (0.35, 0.75)
HUG_WINDOW_ABS_Y = 0.15
MAX_STATION_NAV_M = 2.60
APPROACH_STALL_SEC = 4.0


def validate_yellow_world(box_world) -> np.ndarray:
    """Reject detections that are not the fixed-layout cube-top yellow cell."""
    box = np.asarray(box_world, dtype=float)
    if box.shape != (3,) or not np.all(np.isfinite(box)):
        raise ValueError("yellow box_world must be a finite [x, y, z] vector")
    if not (-0.80 <= box[0] <= -0.30 and 2.05 <= box[1] <= 2.55 and 0.94 <= box[2] <= 1.07):
        raise ValueError(f"yellow box is outside the Task 3 cube-top window: {box.tolist()}")
    return box


def snap_cube_top_center(box_world) -> np.ndarray:
    """Snap X/Z to the cube-top cell; pull measured Y only a little toward layout.

    Full snap to y=2.30 puts the palms 4 cm behind a south-biased reconstruction,
    so the arm midpoint is on the far half instead of the object center.
    Depth is typically short, so allow at most 2 cm of north pull.
    """
    box = validate_yellow_world(box_world).copy()
    box[0] = YELLOW_FIXED_WORLD[0]
    box[2] = CUBE_TOP_CENTER_Z
    layout_y = float(YELLOW_FIXED_WORLD[1])
    pull = float(np.clip(layout_y - float(box[1]), -MAX_HUG_Y_PULL_M, MAX_HUG_Y_PULL_M))
    box[1] = float(box[1]) + pull
    max_y = STATION_Y_MAX + TASK3_STANDOFF_M - 0.01
    if float(box[1]) > max_y:
        box[1] = float(max_y)
    return box


def center_from_cube_surface(
    surface_world,
    yaw: float = GRASP_YAW,
    half_depth: float = BOX_HALF_DEPTH,
    center_z: float = CUBE_TOP_CENTER_Z,
) -> np.ndarray:
    """Map an RGB-D hit on the cube-top box front face to its center."""
    surface = np.asarray(surface_world, dtype=float)
    yaw = wrap_to_pi(yaw)
    depth = float(half_depth)
    height = float(center_z)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError("surface must be a finite 3-vector")
    if not 0.05 <= depth <= 0.14:
        raise ValueError("cube-top half-depth must be within [0.05, 0.14] m")
    center = surface.copy()
    center[:2] += depth * np.array([math.cos(yaw), math.sin(yaw)])
    center[2] = height
    return center


def validate_place_world_l1(place_world) -> np.ndarray:
    """Reject place targets that are not the fixed-layout Task 3 L1 cell."""
    place = np.asarray(place_world, dtype=float)
    if place.shape != (3,) or not np.all(np.isfinite(place)):
        raise ValueError("place_world must be a finite [x, y, z] vector")
    if not (-2.95 <= place[0] <= -2.40 and 0.30 <= place[1] <= 0.72 and 0.42 <= place[2] <= 0.60):
        raise ValueError(f"place_world is outside the Task 3 shelf L1 window: {place.tolist()}")
    return place


def held_center_from_palms_l1(slide: float, left_joints, right_joints) -> np.ndarray:
    """Palm midpoint at Task 3 L1 height; the Task 2 helper rejects z below 0.70 m."""
    backend = MMK2KdlBackend()
    left_fk = backend.forward("l", float(slide), np.asarray(left_joints, dtype=float))
    right_fk = backend.forward("r", float(slide), np.asarray(right_joints, dtype=float))
    center = 0.5 * (np.asarray(left_fk[:3, 3], dtype=float) + np.asarray(right_fk[:3, 3], dtype=float))
    center[0] -= TASK3_GRASP_FWD_OFFSET_M
    if not (0.35 <= center[0] <= 0.85 and abs(center[1]) <= 0.18 and 0.45 <= center[2] <= 1.20):
        raise ValueError(f"palm midpoint is outside the L1 hold window: {center.tolist()}")
    return center


def lift_clears_cube(box_z: float, lift_height: float, cube_top: float = CUBE_TOP_Z) -> bool:
    """True when a lifted cube-top box bottom clears the cube top with margin."""
    bottom_after_lift = float(box_z) - BOX_HALF_HEIGHT_M + float(lift_height)
    return bottom_after_lift >= float(cube_top) + 0.04


def held_bottom_z(held_center_z, half_height: float = BOX_HALF_HEIGHT_M) -> float:
    """World-z of the held carton bottom from its center height."""
    return float(held_center_z) - float(half_height)


def held_top_z(held_center_z, half_height: float = BOX_HALF_HEIGHT_M) -> float:
    """World-z of the held carton top from its center height."""
    return float(held_center_z) + float(half_height)


def cubby_fits_l1(held_center_z) -> bool:
    """True when the held carton fits in the L1 opening under the L2 board."""
    bottom = held_bottom_z(held_center_z)
    top = held_top_z(held_center_z)
    return (
        bottom >= SHELF_L1_BOARD_Z + L1_BOARD_CLEARANCE_M
        and top <= SHELF_L2_BOARD_Z - L2_CLEARANCE_M
    )


def approach_clears_packaging(
    held_center_z,
    packaging_top: float = PACKAGING_TOP_Z_M,
) -> bool:
    """True when a height would fly over the L1 packaging. The L1 cubby cannot."""
    return held_bottom_z(held_center_z) >= float(packaging_top) + 0.04


def place_is_l1_layer(place_z, packaging_z: float = float(PACKAGING_WORLD[2])) -> bool:
    """True when the place height is on the same board as the packaging box."""
    return abs(float(place_z) - float(packaging_z)) <= 0.05


def nearest_shelf_board_z(object_z, half_height: float = PACKAGING_HALF_HEIGHT_M) -> float:
    """Shelf board the detected obstacle is sitting on."""
    bottom = float(object_z) - float(half_height)
    return float(min(SHELF_BOARD_SURFACES_Z, key=lambda board: abs(float(board) - bottom)))


def validate_shelf_side_place(place_world) -> np.ndarray:
    """Accept a shelf cell beside a detected prop, not only the fixed L1 numbers."""
    place = np.asarray(place_world, dtype=float)
    if place.shape != (3,) or not np.all(np.isfinite(place)):
        raise ValueError("place_world must be a finite [x, y, z] vector")
    if not (-2.95 <= place[0] <= -2.35 and 0.22 <= place[1] <= 1.15 and 0.40 <= place[2] <= 1.25):
        raise ValueError(f"place_world is outside the shelf side-place window: {place.tolist()}")
    return place


def l1_clear_bay_y(
    obstacle_y,
    *,
    direction: str = "left",
    obstacle_half_y: float = OBSTACLE_HALF_Y_M,
    aisle_y: float = SHELF_AISLE_Y_M,
    post_span_y: float = SHELF_POST_LOCAL_Y_M,
    post_half: float = SHELF_POST_HALF_M,
) -> float:
    """Mid-Y of the free gap between a corner post and the packaging box."""
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'")
    aisle_y = float(aisle_y)
    if direction == "left":
        post_inner = aisle_y - float(post_span_y) + float(post_half)
        obs_face = float(obstacle_y) - float(obstacle_half_y)
    else:
        post_inner = aisle_y + float(post_span_y) - float(post_half)
        obs_face = float(obstacle_y) + float(obstacle_half_y)
    return 0.5 * (post_inner + obs_face)


def place_left_of_obstacle(
    obstacle_world,
    *,
    direction: str = "left",
    box_half_y: float = BOX_HALF_WIDTH_M,
    box_half_z: float = BOX_HALF_HEIGHT_M,
    obstacle_half_y: float = OBSTACLE_HALF_Y_M,
    gap: float = PLACE_LEFT_GAP_M,
    inset: float = PLACE_SHELF_INSET_M,
) -> np.ndarray:
    """World center in the open bay left of the obstacle, same shelf layer.

    The robot faces west, so left is south (-Y).  Y is the midpoint between
    the south post's inner face and the packaging's south face, not a tight
    squeeze against the post.  Height comes from the board under the obstacle.
    """
    del gap, box_half_y
    obs = np.asarray(obstacle_world, dtype=float)
    if obs.shape != (3,) or not np.all(np.isfinite(obs)):
        raise ValueError("obstacle_world must be a finite 3-vector")
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'")
    board = nearest_shelf_board_z(obs[2])
    place = np.array([
        float(obs[0]) - float(inset),
        l1_clear_bay_y(obs[1], direction=direction, obstacle_half_y=obstacle_half_y),
        float(board) + float(box_half_z),
    ], dtype=float)
    return validate_shelf_side_place(place)


def approach_z_over_obstacle(obstacle_z, half_height: float = PACKAGING_HALF_HEIGHT_M) -> float:
    """Held-center height for L1 insert. A 0.19 m box cannot fly over the obstacle under L2."""
    del obstacle_z, half_height
    return float(TASK3_APPROACH_Z_M)


def south_then_west_insert_plan(
    start_xy,
    place_stand_xy,
    *,
    place_yaw: float = PLACE_YAW,
    south_shift_min: float = SOUTH_SHIFT_MIN_M,
) -> dict:
    """South in front of the cabinet, then west into the L1 cell under L2.

    West along the aisle at z=0.82 puts the held bottom on the L2 board.
    Shift south at the current x first, then insert west left of the packaging.
    """
    start = np.asarray(start_xy, dtype=float)[:2]
    stand = np.asarray(place_stand_xy, dtype=float)[:2]
    if start.shape != (2,) or stand.shape != (2,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(stand)):
        raise ValueError("south-then-west ends must be finite xy")
    south_xy = np.array([float(start[0]), float(stand[1])], dtype=float)
    shift_m = abs(float(stand[1]) - float(start[1]))
    south_bearing = -math.pi / 2.0 if float(stand[1]) <= float(start[1]) else math.pi / 2.0
    return {
        "south_xy": south_xy,
        "place_stand_xy": np.array([float(stand[0]), float(stand[1])], dtype=float),
        "west_yaw": wrap_to_pi(place_yaw),
        "south_bearing": south_bearing,
        "needs_south_shift": shift_m >= float(south_shift_min),
        "south_shift_m": shift_m,
    }


def snap_packaging_center(obstacle_world) -> np.ndarray:
    """Keep a nearby white-box lock; otherwise use the fixed-layout packaging pose."""
    obs = np.asarray(obstacle_world, dtype=float)
    if obs.shape != (3,) or not np.all(np.isfinite(obs)):
        raise ValueError("obstacle_world must be a finite 3-vector")
    if float(np.linalg.norm(obs[:2] - PACKAGING_WORLD[:2])) <= 0.25:
        snapped = PACKAGING_WORLD.copy()
        return snapped
    return obs.copy()


def station_for_yellow(
    box_world,
    standoff: float = TASK3_STANDOFF_M,
    yaw: float = GRASP_YAW,
) -> np.ndarray:
    """South-of-table station that puts the cube-top box into the hug window."""
    stand = station_target(box_world, standoff, yaw).copy()
    if float(stand[1]) > TABLE_SOUTH_EDGE_Y - 0.01:
        raise ValueError(f"station {stand.tolist()} is not south of the table edge")
    if float(stand[1]) > STATION_Y_MAX:
        stand[1] = STATION_Y_MAX
    return stand


def place_stand_from_goal(place_world, place_yaw: float, held_center_base) -> np.ndarray:
    """Base xy that puts the held box center onto place_world xy at place_yaw."""
    place = validate_shelf_side_place(place_world)
    held = np.asarray(held_center_base, dtype=float)
    yaw = wrap_to_pi(place_yaw)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_center_base must be a finite 3-vector")
    if not 0.30 <= held[0] <= 0.80 or abs(held[1]) > 0.18:
        raise ValueError(f"held box is outside the safe base-frame window: {held.tolist()}")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    held_world = np.array([
        cosine * held[0] - sine * held[1],
        sine * held[0] + cosine * held[1],
    ])
    return np.array([place[0] - held_world[0], place[1] - held_world[1]], dtype=float)


def aisle_staging_from_stand(
    place_stand_xy,
    place_yaw: float,
    staging_back: float = STAGING_BACK_M,
    aisle_y: float = SHELF_AISLE_Y_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage on the proven shelf-center line instead of the south L1 cell."""
    stand = np.asarray(place_stand_xy, dtype=float)
    aisle_y = float(aisle_y)
    if stand.shape != (2,) or not np.all(np.isfinite(stand)):
        raise ValueError("place stand must be a finite [x, y] vector")
    if not 0.70 <= aisle_y <= 0.86:
        raise ValueError(f"shelf aisle y {aisle_y} is outside the proven insert line")
    insert_xy = np.array([float(stand[0]), aisle_y], dtype=float)
    return staging_pose(insert_xy, place_yaw, staging_back), insert_xy


def aligned_shelf_insert_plan(
    place_stand_xy,
    *,
    aisle_y: float = SHELF_AISLE_Y_M,
    place_yaw: float = PLACE_YAW,
    south_shift_min: float = SOUTH_SHIFT_MIN_M,
) -> dict:
    """West along the aisle to insert x, then south to the L1 cell, then square west."""
    stand = np.asarray(place_stand_xy, dtype=float)
    aisle_y = float(aisle_y)
    if stand.shape != (2,) or not np.all(np.isfinite(stand)):
        raise ValueError("place stand must be a finite [x, y] vector")
    if not 0.70 <= aisle_y <= 0.86:
        raise ValueError(f"shelf aisle y {aisle_y} is outside the proven insert line")
    insert_xy = np.array([float(stand[0]), aisle_y], dtype=float)
    shift_m = abs(float(stand[1]) - aisle_y)
    south_bearing = None
    if shift_m >= float(south_shift_min):
        south_bearing = approach_bearing(insert_xy, stand)
    return {
        "insert_xy": insert_xy,
        "place_stand_xy": np.array([float(stand[0]), float(stand[1])], dtype=float),
        "west_yaw": wrap_to_pi(place_yaw),
        "south_bearing": south_bearing,
        "needs_south_shift": south_bearing is not None,
        "south_shift_m": shift_m,
    }


def insert_line_y_at_x(start_xy, goal_xy, query_x: float) -> float:
    """World y of the straight insert segment at a given x."""
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    if start.shape != (2,) or goal.shape != (2,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("insert segment ends must be finite xy")
    dx = float(goal[0] - start[0])
    if abs(dx) < 1e-6:
        return float(start[1])
    t = (float(query_x) - float(start[0])) / dx
    return float(start[1] + t * (goal[1] - start[1]))


def approach_bearing(start_xy, goal_xy) -> float:
    """Heading of the chassis-forward insert from start xy to the place stand."""
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    if start.shape != (2,) or goal.shape != (2,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("approach segment ends must be finite xy")
    delta = goal - start
    if float(np.linalg.norm(delta)) < 0.08:
        raise ValueError(f"approach segment is too short: {delta.tolist()}")
    return math.atan2(float(delta[1]), float(delta[0]))


def palm_pair_xyz(slide: float, left_joints, right_joints) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame palm origins for a locked hug."""
    backend = MMK2KdlBackend()
    left_fk = backend.forward("l", float(slide), np.asarray(left_joints, dtype=float))
    right_fk = backend.forward("r", float(slide), np.asarray(right_joints, dtype=float))
    return np.asarray(left_fk[:3, 3], dtype=float), np.asarray(right_fk[:3, 3], dtype=float)


def hold_palm_metrics(slide: float, left_joints, right_joints) -> dict:
    """Palm span/tilt used to catch a one-sided sag before carry."""
    left_p, right_p = palm_pair_xyz(slide, left_joints, right_joints)
    half = float(0.5 * (left_p[1] - right_p[1]))
    return {
        "left_xyz": left_p.tolist(),
        "right_xyz": right_p.tolist(),
        "dz_m": float(left_p[2] - right_p[2]),
        "dx_m": float(left_p[0] - right_p[0]),
        "half_span_m": half,
        "mid_xyz": (0.5 * (left_p + right_p)).tolist(),
        "level_enough": (
            abs(float(left_p[2] - right_p[2])) <= MAX_HOLD_PALM_DZ_M
            and MIN_HOLD_HALF_SPAN_M <= half <= MAX_HOLD_HALF_SPAN_M
        ),
    }


def level_hold_pose(slide: float, left_joints, right_joints, hold_half: float = TASK1_HOLD_HALF_M) -> dict:
    """Re-solve a same-height 0.115 m hug around the current palm midpoint."""
    half = float(hold_half)
    if not MIN_HOLD_HALF_SPAN_M <= half <= TASK1_APPROACH_HALF_M:
        raise ValueError(f"level hold half {half} is outside the side-face window")
    left_p, right_p = palm_pair_xyz(slide, left_joints, right_joints)
    mid = 0.5 * (left_p + right_p)
    left_target = np.array([mid[0], mid[1] + half, mid[2]], dtype=float)
    right_target = np.array([mid[0], mid[1] - half, mid[2]], dtype=float)
    return solve_bimanual_pose(slide, left_joints, right_joints, left_target, right_target)


def _bind_capped_hold_keeper(context, result, max_refresh: int = MAX_HOLD_REFRESH):
    """Keep at most one extra j1 squeeze so palms do not crawl onto the box top."""
    max_refresh = int(max_refresh)
    if max_refresh < 0:
        raise ValueError("max hold refresh must be >= 0")
    inner = _bind_hold_keeper(context, result)

    def keeper(left_current, right_current, left_hold, right_hold):
        if int(result.get("hold_refresh_count", 0)) >= max_refresh:
            contact = carry_hold_ok(left_current, right_current, left_hold, right_hold)
            return np.asarray(left_hold, dtype=float), np.asarray(right_hold, dtype=float), contact
        return inner(left_current, right_current, left_hold, right_hold)

    return keeper


def task3_placement_error(held_world, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Local xy/z error of the locked box versus the instruction place point."""
    held = np.asarray(held_world, dtype=float)
    place = validate_shelf_side_place(place_world)
    radius = float(place_radius)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_world must be a finite 3-vector")
    if not 0.06 <= radius <= 0.30:
        raise ValueError("place radius must be within [0.06, 0.30] m")
    xy_error = float(np.linalg.norm(held[:2] - place[:2]))
    z_error = float(abs(held[2] - place[2]))
    return {
        "xy_error_m": xy_error,
        "z_error_m": z_error,
        "within_radius": xy_error <= radius,
    }


def box_inside_place_radius(base_pose, held_center_base, place_world, place_radius: float = PLACE_RADIUS_M) -> dict:
    """Judge the locked box xy against the instruction place cylinder."""
    world = held_center_world(base_pose, held_center_base)
    error = task3_placement_error(world, place_world, place_radius)
    error["held_world"] = world.tolist()
    return error


def shelf_inward_ok(held_world, place_world, max_outward_m: float = TASK3_MAX_PLACE_OUTWARD_M) -> dict:
    """True when the box is not still hanging off the west-facing shelf lip."""
    held = np.asarray(held_world, dtype=float)
    place = validate_shelf_side_place(place_world)
    limit = float(max_outward_m)
    if held.shape != (3,) or not np.all(np.isfinite(held)):
        raise ValueError("held_world must be a finite 3-vector")
    if not 0.02 <= limit <= 0.18:
        raise ValueError("max outward offset must be within [0.02, 0.18] m")
    outward = float(held[0] - place[0])
    return {
        "outward_m": outward,
        "deep_enough": outward <= limit,
    }


def local_release_open(hold_left, hold_right, open_rad: float = TASK3_RELEASE_OPEN_RAD):
    """Open the hug at j2 without a new Cartesian solve that can flare into the posts."""
    left = np.asarray(hold_left, dtype=float).copy()
    right = np.asarray(hold_right, dtype=float).copy()
    open_rad = float(open_rad)
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError("release-open vectors must be six-joint arrays")
    if not 0.04 <= open_rad <= 0.12:
        raise ValueError("release-open must stay a small unpinch, not a full arm spread")
    left[1] += open_rad
    right[1] += open_rad
    return left, right


def l1_release_cartesian(
    left_position,
    right_position,
    *,
    spread: float = TASK3_RELEASE_SPREAD_M,
    withdraw: float = TASK3_RELEASE_WITHDRAW_M,
):
    """Open the L1 hug wider than the carton and pull the fingertips off the wrap."""
    left = np.asarray(left_position, dtype=float)
    right = np.asarray(right_position, dtype=float)
    spread = float(spread)
    withdraw = float(withdraw)
    if left.shape != (3,) or right.shape != (3,) or not np.all(np.isfinite([left, right])):
        raise ValueError("release Cartesian targets must be finite 3-vectors")
    if not 0.03 <= spread <= 0.06:
        raise ValueError("L1 release spread must stay inside the post/packaging bay")
    if not 0.02 <= withdraw <= 0.06:
        raise ValueError("L1 release withdraw is out of range")
    return (
        left + np.array([-withdraw, spread, 0.0]),
        right + np.array([-withdraw, -spread, 0.0]),
    )


def l1_release_joints(slide, hold_left, hold_right, spread: float = TASK3_RELEASE_SPREAD_M):
    """Unpinch in the L1 bay. Prefer spread+withdraw; fall back to Y-only, then j2."""
    backend = MMK2KdlBackend()
    hold_left = np.asarray(hold_left, dtype=float)
    hold_right = np.asarray(hold_right, dtype=float)
    left_fk = backend.forward("l", float(slide), hold_left)
    right_fk = backend.forward("r", float(slide), hold_right)
    attempts = (
        ("cartesian_withdraw", lambda: l1_release_cartesian(left_fk[:3, 3], right_fk[:3, 3], spread=spread)),
        ("cartesian", lambda: release_cartesian(left_fk[:3, 3], right_fk[:3, 3], spread)),
    )
    for how, builder in attempts:
        try:
            left_xyz, right_xyz = builder()
            plan = solve_bimanual_pose(
                float(slide), hold_left, hold_right, left_xyz, right_xyz, backend=backend,
            )
            return (
                np.asarray(plan["left_joint_target"], dtype=float),
                np.asarray(plan["right_joint_target"], dtype=float),
                how,
                plan,
            )
        except (ValueError, RuntimeError):
            continue
    left, right = local_release_open(hold_left, hold_right)
    return left, right, "joint_open", None


def locate_packaging(node, pick_yaw: float = PLACE_YAW):
    """RGB-D lock of the shelf packaging box, mapped to its center."""
    snapshot = node.wait_for_snapshot(timeout_sec=4.0)
    detections = detect_colored_boxes(
        snapshot.rgb,
        "white",
        min_area=max(40, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 8000),
        max_area_frac=0.20,
        max_bbox_frac=0.28,
    )
    if not detections:
        raise RuntimeError("no white packaging box detected in the current RGB frame")
    detection = max(detections, key=lambda item: item.area * item.confidence)
    depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
    camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
    frame = snapshot.camera_frame or "head_camera"
    camera_to_world = node.transforms.lookup("odom", frame)
    surface_world = transform_point(camera_to_world, camera_point)
    center = snap_packaging_center(center_from_shelf_front(surface_world, pick_yaw, half_depth=OBSTACLE_HALF_X_M))
    if not (-2.95 <= center[0] <= -2.35 and 0.30 <= center[1] <= 1.20 and 0.42 <= center[2] <= 1.25):
        raise RuntimeError(f"white detection is not a shelf packaging box: {center.tolist()}")
    center_base = _world_to_base(node, center)
    return {
        "bbox": list(detection.bbox),
        "pixel": list(detection.center),
        "depth_m": float(depth),
        "surface_world": surface_world.tolist(),
        "center_world": center.tolist(),
        "center_base": center_base.tolist(),
        "source": "vision",
    }


def _look_at_shelf_obstacle(node, result):
    node.controller.command_head(list(SHELF_OBSERVE_HEAD))
    result["published_control_topics"] = list(dict.fromkeys(
        result.get("published_control_topics", []) + ["/head_forward_position_controller/commands"]
    ))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        node.spin_once(0.05)


def locate_yellow(node, yaw: float = GRASP_YAW):
    """RGB-D lock of the cube-top yellow front face, mapped to the box center."""
    snapshot = node.wait_for_snapshot(timeout_sec=4.0)
    detections = detect_colored_boxes(
        snapshot.rgb,
        TASK_COLOR,
        min_area=max(60, snapshot.rgb.shape[0] * snapshot.rgb.shape[1] // 5000),
    )
    if not detections:
        raise RuntimeError("no yellow box detected in the current RGB frame")
    detection = max(detections, key=lambda item: item.area * item.confidence)
    depth = robust_depth_from_bbox(snapshot.depth_m, detection.bbox, min_samples=5)
    camera_point = snapshot.intrinsics.project_pixel(*detection.center, depth)
    frame = snapshot.camera_frame or "head_camera"
    camera_to_world = node.transforms.lookup("odom", frame)
    surface_world = transform_point(camera_to_world, camera_point)
    center_raw = validate_yellow_world(center_from_cube_surface(surface_world, yaw))
    center_world = snap_cube_top_center(center_raw)
    center_base = _world_to_base(node, center_world)
    return {
        "bbox": list(detection.bbox),
        "pixel": list(detection.center),
        "depth_m": float(depth),
        "surface_world": surface_world.tolist(),
        "center_world_raw": center_raw.tolist(),
        "center_world": center_world.tolist(),
        "center_base": center_base.tolist(),
        "source": "vision",
    }


def _look_at_table(node, result):
    node.controller.command_head(list(OBSERVE_HEAD))
    result["published_control_topics"] = list(dict.fromkeys(
        result.get("published_control_topics", []) + ["/head_forward_position_controller/commands"]
    ))
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        node.spin_once(0.05)


def _record_topic(result, topic: str) -> None:
    result["published_control_topics"] = list(dict.fromkeys(result.get("published_control_topics", []) + [topic]))


def _drive_line_task3(
    node,
    start_pose,
    target_pose,
    direction: int,
    timeout: float,
    result,
    left_joints,
    right_joints,
    gripper_open: float,
    *,
    require_hold: bool = True,
    position_tolerance: float = 0.02,
    min_traveled_m: float = 0.0,
    max_linear_speed: float = TASK3_SHELF_LINEAR_SPEED,
    max_angular_speed: float = TASK3_LINE_ANGULAR_SPEED,
    key_prefix: str = "line",
    held_center_base=None,
    place_world=None,
    place_radius: float | None = None,
    hold_keeper=None,
):
    """Heading-aligned segment with the Task 3 L1 place accept checks."""
    from task1_shelf_place_check import held_line_command

    deadline = time.monotonic() + float(timeout)
    final = None
    max_cross_track = 0.0
    last_progress_at = time.monotonic()
    last_traveled = -1.0
    left_joints = np.asarray(left_joints, dtype=float)
    right_joints = np.asarray(right_joints, dtype=float)
    while time.monotonic() < deadline:
        node.spin_once(0.05)
        left_current, right_current, left_gripper, right_gripper = _current_arm_state_unbounded(node)
        samples = result.setdefault(f"{key_prefix}_raw_gripper_feedback", [])
        if len(samples) < 20:
            samples.append({"left": float(left_gripper), "right": float(right_gripper)})
        if not (0.0 <= left_gripper <= 1.0 and 0.0 <= right_gripper <= 1.0):
            result[f"{key_prefix}_gripper_endpoint_warning"] = True
        if hold_keeper is not None:
            left_joints, right_joints, contact = hold_keeper(
                left_current, right_current, left_joints, right_joints,
            )
            _command_hug(node, left_joints, right_joints, gripper_open)
            result[f"{key_prefix}_contact_feedback"] = contact
            if require_hold and not contact["holding"]:
                node.controller.stop_base()
                raise RuntimeError(f"held-box contact changed during {key_prefix}: {contact}")
        elif require_hold:
            _command_hug(node, left_joints, right_joints, gripper_open)
            contact = carry_hold_ok(left_current, right_current, left_joints, right_joints)
            result[f"{key_prefix}_contact_feedback"] = contact
            if not contact["holding"]:
                node.controller.stop_base()
                raise RuntimeError(f"held-box contact changed during {key_prefix}: {contact}")
        else:
            _command_hug(node, left_joints, right_joints, gripper_open)
        current = _odom_pose(node)
        linear, angular, details = held_line_command(
            current, start_pose, target_pose, direction, position_tolerance, 0.05,
            min_traveled_m=min_traveled_m, max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )
        if held_center_base is not None and place_world is not None and place_radius is not None:
            inside = box_inside_place_radius(current, held_center_base, place_world, place_radius)
            depth = shelf_inward_ok(inside["held_world"], place_world)
            result[f"{key_prefix}_estimated_place_world"] = inside["held_world"]
            result[f"{key_prefix}_xy_error_m"] = inside["xy_error_m"]
            result[f"{key_prefix}_outward_m"] = depth["outward_m"]
            if (
                inside["within_radius"]
                and depth["deep_enough"]
                and details["remaining_m"] <= max(float(position_tolerance), TASK3_PLACE_REMAINING_OK_M)
            ):
                node.controller.stop_base()
                result[f"{key_prefix}_accepted_inside_radius"] = True
                result.update({
                    f"{key_prefix}_phase": details["phase"],
                    f"{key_prefix}_remaining_m": details["remaining_m"],
                    f"{key_prefix}_traveled_m": details["traveled_m"],
                    f"{key_prefix}_cross_track_m": details["cross_track_m"],
                    f"{key_prefix}_yaw_error_rad": details["yaw_error_rad"],
                })
                return current
        max_cross_track = max(max_cross_track, details["cross_track_m"])
        final = current
        result.update({
            f"{key_prefix}_phase": details["phase"],
            f"{key_prefix}_remaining_m": details["remaining_m"],
            f"{key_prefix}_traveled_m": details["traveled_m"],
            f"{key_prefix}_cross_track_m": details["cross_track_m"],
            f"{key_prefix}_yaw_error_rad": details["yaw_error_rad"],
            f"{key_prefix}_max_cross_track_m": max_cross_track,
        })
        if details["traveled_m"] >= last_traveled + 0.01:
            last_traveled = details["traveled_m"]
            last_progress_at = time.monotonic()
        elif (
            place_world is not None
            and time.monotonic() - last_progress_at >= APPROACH_STALL_SEC
        ):
            node.controller.stop_base()
            raise TimeoutError(
                f"{key_prefix} stalled after {details['traveled_m']:.3f} m; "
                f"remaining={details['remaining_m']:.3f} m; final={current.tolist()}"
            )
        if details["phase"] == "complete":
            node.controller.stop_base()
            return final
        node.controller.publish_velocity(linear, angular)
        _record_topic(result, "/cmd_vel")
    node.controller.stop_base()
    raise TimeoutError(f"{key_prefix} timed out; final={None if final is None else final.tolist()}")


def _establish_cube_top_hold(node, args, box_base, result):
    """Reproduce the verified Task 1 hug at the raised cube-top height."""
    initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
    initial_left_gripper = command_gripper_value(raw_left_gripper)
    initial_right_gripper = command_gripper_value(raw_right_gripper)
    initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
    if initial_slide > 0.10:
        raise RuntimeError("slide must start at a safe high posture (<= 0.10 m)")
    # Palm Z is locked to PRE_GRASP_Z0 - slide with Task 1's wrist rotation.
    # Raising the torso without changing that rotation yaws j0 and twists the hug.
    contact_slide = float(PRE_GRASP_Z0 - (box_base[2] + args.grasp_z_offset))
    if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
        raise RuntimeError(f"contact slide {contact_slide:.4f} is outside {SLIDE_LIMITS}")
    lift_slide = lift_slide_target(contact_slide, args.lift_height)
    if not lift_clears_cube(float(box_base[2]), args.lift_height):
        raise RuntimeError("lift height does not clear the white cube top")
    high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
    plans = []
    left_reference = np.asarray(high_plan["left_joint_target"])
    right_reference = np.asarray(high_plan["right_joint_target"])
    clearances = contact_clearance_schedule(args.initial_clearance, args.contact_step)
    for clearance in clearances:
        half = args.hold_half if clearance == 0.0 else args.approach_half
        left_target, right_target = contact_approach_geometry(
            box_base, clearance, args.grasp_fwd_offset, args.grasp_z_offset, half,
        )
        plan = solve_bimanual_pose(contact_slide, left_reference, right_reference, left_target, right_target)
        plan["clearance_m"] = clearance
        plan["half_gap_m"] = half
        plans.append(plan)
        left_reference = np.asarray(plan["left_joint_target"])
        right_reference = np.asarray(plan["right_joint_target"])
    result.update({
        "initial_slide": initial_slide,
        "initial_raw_gripper_feedback": {"left": raw_left_gripper, "right": raw_right_gripper},
        "contact_slide": contact_slide,
        "lift_slide": lift_slide,
        "high_pregrasp_plan": high_plan,
        "contact_plans": plans,
        "contact_clearance_schedule_m": clearances,
    })
    context = {
        "initial_left": initial_left,
        "initial_right": initial_right,
        "initial_left_gripper": initial_left_gripper,
        "initial_right_gripper": initial_right_gripper,
        "initial_slide": initial_slide,
        "contact_slide": contact_slide,
        "lift_slide": lift_slide,
        "high_plan": high_plan,
        "plans": plans,
        "hold_left": None,
        "hold_right": None,
        "held_center_base": None,
    }
    high_left, high_right = _traverse_pair(
        node, initial_left, initial_right, high_plan["left_joint_target"], high_plan["right_joint_target"],
        args.joint_max_step, args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD,
        initial_left_gripper, initial_right_gripper, True, result,
    )
    context["high_left"], context["high_right"] = high_left, high_right
    open_left, open_right = _traverse_grippers(
        node, high_left, high_right, initial_left_gripper, initial_right_gripper,
        args.gripper_open, args.gripper_open, args.gripper_max_step, args.settle_timeout, 0.010, result,
    )
    result["reached_open_left_gripper"] = open_left
    result["reached_open_right_gripper"] = open_right
    # Arms stay at the wide open pose while the slide drops to cube-top height.
    _traverse_spine_keeping_pose(
        node, initial_slide, contact_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        high_left, high_right, args.gripper_open, result,
    )
    reached_left, reached_right = high_left, high_right
    reached = []
    contact_plan = None
    contact_feedback = None
    hold_left = hold_right = None
    for plan_index, plan in enumerate(plans):
        result["phase"] = f"approach_clearance_{plan['clearance_m']:.3f}"
        try:
            reached_left, reached_right, waypoint_result = _approach_until_reached_or_contact(
                node, reached_left, reached_right, plan["left_joint_target"], plan["right_joint_target"],
                args.joint_max_step, args.gripper_open, args.gripper_open,
                args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD, result,
                allow_early_contact=plan["clearance_m"] <= CARRY_CLEARANCE_MAX_M,
            )
        except TimeoutError as exc:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            locked = blocked_table_hug_lock(
                left_now, right_now, high_left, high_right,
                plan["left_joint_target"], plan["right_joint_target"],
            )
            if locked is None:
                raise
            reached_left, reached_right = locked["left"], locked["right"]
            waypoint_result = locked["feedback"]
            result["blocked_hug_lock"] = {
                "clearance_m": plan["clearance_m"],
                "left": reached_left.tolist(),
                "right": reached_right.tolist(),
                "timeout_error": str(exc),
                "left_max_joint_residual_rad": waypoint_result["left_max_joint_residual_rad"],
                "right_max_joint_residual_rad": waypoint_result["right_max_joint_residual_rad"],
            }
        reached.append({
            "clearance_m": plan["clearance_m"],
            "left": np.asarray(reached_left, dtype=float).tolist(),
            "right": np.asarray(reached_right, dtype=float).tolist(),
            "contact_detected": bool(waypoint_result.get("contact_detected")),
            "blocked_hug": bool(waypoint_result.get("blocked_hug")),
        })
        if not waypoint_result.get("contact_detected"):
            continue
        if not hug_moved_from_pregrasp(reached_left, reached_right, high_left, high_right):
            result.setdefault("rejected_false_contacts", []).append({
                "clearance_m": plan["clearance_m"], "reason": "still_at_pregrasp",
            })
            continue
        if plan["clearance_m"] > CARRY_CLEARANCE_MAX_M and not waypoint_result.get("blocked_hug"):
            result.setdefault("rejected_false_contacts", []).append({
                "clearance_m": plan["clearance_m"], "reason": "too_open_to_carry",
            })
            continue
        # The 0.01 m waypoint still uses the 0.13 m approach half-gap.  Do not
        # lock that open pose, and do not stop 22% of the way to 0.115 m: the
        # hold keeper would then keep tightening j1 and crawl onto the box top.
        tight = plans[-1]
        if plan["clearance_m"] > 0.0:
            result["phase"] = "close_to_hold_gap"
            try:
                reached_left, reached_right, close_result = _approach_until_reached_or_contact(
                    node, reached_left, reached_right, tight["left_joint_target"], tight["right_joint_target"],
                    args.joint_max_step, args.gripper_open, args.gripper_open,
                    args.settle_timeout, APPROACH_JOINT_TOLERANCE_RAD, result,
                    allow_early_contact=True,
                )
                result["close_to_hold_gap"] = {
                    "contact_detected": bool(close_result.get("contact_detected")),
                    "blocked_hug": bool(close_result.get("blocked_hug")),
                    "left_max_joint_residual_rad": close_result.get("left_max_joint_residual_rad"),
                    "right_max_joint_residual_rad": close_result.get("right_max_joint_residual_rad"),
                }
            except TimeoutError as exc:
                left_now, right_now, _, _ = _current_arm_state_unbounded(node)
                locked = blocked_table_hug_lock(
                    left_now, right_now, high_left, high_right,
                    tight["left_joint_target"], tight["right_joint_target"],
                )
                if locked is None:
                    raise
                reached_left, reached_right = locked["left"], locked["right"]
                result["close_to_hold_gap_blocked"] = True
                result["close_to_hold_gap_timeout_error"] = str(exc)
        hold_left, hold_right = local_carry_hold(
            reached_left, reached_right, TASK3_HOLD_SQUEEZE_RAD,
        )
        try:
            squeeze = _confirm_inward_squeeze(
                node, hold_left, hold_right, args.gripper_open, args.squeeze_seconds, result,
            )
            result["squeeze_confirmed"] = True
            result["squeeze_feedback"] = squeeze
        except RuntimeError as exc:
            empty_close = "without the box blocking" in str(exc)
            if waypoint_result.get("blocked_hug") and not empty_close:
                result["blocked_hug_squeeze_error"] = str(exc)
            else:
                result.setdefault("rejected_false_contacts", []).append({
                    "clearance_m": plan["clearance_m"], "error": str(exc),
                    "feedback": result.get("squeeze_feedback"),
                })
                if empty_close:
                    result["empty_hold_close"] = True
                    break
                reached_left, reached_right, _, _ = _current_arm_state_unbounded(node)
                continue
        contact_plan = plan
        contact_feedback = waypoint_result
        break
    if contact_plan is None:
        if result.get("empty_hold_close"):
            raise TimeoutError(
                "hug closed through empty space; palms were not on the yellow box faces"
            )
        raise TimeoutError("dual-arm contact was not established at any validated clearance waypoint")
    result["contact_detected"] = True
    result["contact_feedback"] = contact_feedback
    result["contact_clearance_detected_m"] = contact_plan["clearance_m"]
    result["reached_contact_plans"] = reached
    result["hold_palms_after_squeeze"] = hold_palm_metrics(contact_slide, hold_left, hold_right)
    mid = np.asarray(result["hold_palms_after_squeeze"]["mid_xyz"], dtype=float)
    result["hug_center_error_m"] = {
        "forward_m": float(mid[0] - box_base[0] - float(args.grasp_fwd_offset)),
        "lateral_m": float(mid[1] - box_base[1]),
        "height_m": float(mid[2] - (float(box_base[2]) + float(args.grasp_z_offset))),
    }
    if abs(result["hold_palms_after_squeeze"]["dz_m"]) > MAX_HOLD_PALM_DZ_M:
        leveled = level_hold_pose(contact_slide, hold_left, hold_right, args.hold_half)
        hold_left, hold_right = local_carry_hold(
            leveled["left_joint_target"], leveled["right_joint_target"], TASK3_HOLD_SQUEEZE_RAD,
        )
        try:
            squeeze = _confirm_inward_squeeze(
                node, hold_left, hold_right, args.gripper_open, args.squeeze_seconds, result,
            )
        except RuntimeError as exc:
            raise TimeoutError(f"side-face hug could not be leveled: {exc}") from exc
        result["hold_leveled"] = True
        result["leveled_squeeze_feedback"] = squeeze
        result["hold_palms_leveled"] = hold_palm_metrics(contact_slide, hold_left, hold_right)
        if abs(result["hold_palms_leveled"]["dz_m"]) > MAX_HOLD_PALM_DZ_M:
            raise TimeoutError(
                "palms are still not level on the yellow box after height correction: "
                f"{result['hold_palms_leveled']}"
            )
    result["hold_joint_targets"] = {"left": np.asarray(hold_left).tolist(), "right": np.asarray(hold_right).tolist()}
    hold_left = np.asarray(hold_left, dtype=float)
    hold_right = np.asarray(hold_right, dtype=float)
    context["hold_left"] = hold_left
    context["hold_right"] = hold_right
    # The cube top is 0.10 m below the lifted box bottom.  Leaving the table
    # at contact height drags the yellow box off the white cube.
    _traverse_spine_holding(
        node, contact_slide, lift_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
        hold_left, hold_right, args.gripper_open, result,
        hold_keeper=_bind_capped_hold_keeper(context, result),
    )
    result["lift_completed"] = True
    result["lift_slide_feedback"] = float(node.sensors.joint_vector(["slide_joint"])[0])
    hold_left = np.asarray(context.get("hold_left", hold_left), dtype=float)
    hold_right = np.asarray(context.get("hold_right", hold_right), dtype=float)
    result["hold_joint_targets"] = {"left": hold_left.tolist(), "right": hold_right.tolist()}
    result["hold_palms_after_lift"] = hold_palm_metrics(
        float(result["lift_slide_feedback"]), hold_left, hold_right,
    )
    context.update({
        "hold_left": hold_left,
        "hold_right": hold_right,
        "held_center_base": apply_slide_keep_hold(box_base, contact_slide, lift_slide),
    })
    return context


def _recover(node, context, args, result, start_base, phase: str, released: bool, place_world):
    try:
        node.controller.stop_base()
    except Exception:
        pass
    holding = context.get("hold_left") is not None and context.get("hold_right") is not None and not released
    current = None
    try:
        current = _odom_pose(node)
    except Exception:
        current = None
    near_shelf = current is not None and float(current[0]) <= SHELF_ZONE_X_MAX
    on_shelf = False
    if holding and near_shelf and context.get("held_center_base") is not None:
        try:
            inside = box_inside_place_radius(
                current, context["held_center_base"], place_world, args.place_radius,
            )
            depth = shelf_inward_ok(inside["held_world"], place_world)
            result["recovery_estimated_place_world"] = inside["held_world"]
            result["recovery_place_xy_error_m"] = inside["xy_error_m"]
            result["recovery_place_outward_m"] = depth["outward_m"]
            on_shelf = bool(inside["within_radius"] and depth["deep_enough"])
        except Exception as exc:
            result["recovery_place_check_error"] = str(exc)
    if holding and near_shelf and on_shelf:
        result["recovery_released_on_shelf"] = True
        try:
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            release_left, release_right, how, plan = l1_release_joints(
                current_slide, context["hold_left"], context["hold_right"], args.release_spread,
            )
            result["recovery_release_mode"] = how
            if plan is not None:
                result["recovery_release_plan"] = plan
            _traverse_pair(
                node, context["hold_left"], context["hold_right"], release_left, release_right,
                args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
                args.gripper_open, args.gripper_open, True, result,
            )
            context["release_left"] = release_left
            context["release_right"] = release_right
            released = True
            holding = False
            _sleep_holding(node, 0.6, release_left, release_right, args.gripper_open, require_hold=False)
            _drive_line_task3(
                node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                args.retreat_timeout, result, release_left, release_right, args.gripper_open,
                require_hold=False, min_traveled_m=0.20, position_tolerance=0.06,
                max_linear_speed=TASK3_HOLD_LINEAR_SPEED,
                key_prefix="recovery_released_retreat",
            )
            result["recovery_released_retreat_completed"] = True
        except Exception as exc:
            result["recovery_released_on_shelf_error"] = str(exc)
            result["recovery_skipped_pull_out"] = True
    elif holding and near_shelf:
        result["recovery_kept_hold"] = True
        try:
            _drive_line_task3(
                node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                args.retreat_timeout, result, context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, min_traveled_m=0.20, position_tolerance=0.06,
                max_linear_speed=TASK3_SHELF_LINEAR_SPEED,
                key_prefix="recovery_held_retreat", hold_keeper=_bind_capped_hold_keeper(context, result),
            )
            result["recovery_held_retreat_completed"] = True
        except Exception as exc:
            result["recovery_held_retreat_error"] = str(exc)
            result["recovery_skipped_pull_out"] = True
            try:
                _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
            except Exception as hug_exc:
                result["recovery_keep_hold_error"] = str(hug_exc)
    elif holding and phase in {"hug_lift", "table_leave"} and start_base is not None and current is not None:
        try:
            _drive_line_task3(
                node, current, start_base, 1, args.table_leave_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, max_linear_speed=TASK3_TABLE_LEAVE_LINEAR_SPEED, key_prefix="recovery_return",
            )
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            _traverse_spine_holding(
                node, current_slide, context["contact_slide"], args.spine_max_step,
                args.settle_timeout, args.spine_tolerance,
                context["hold_left"], context["hold_right"], args.gripper_open, result,
            )
            result["recovery_lowered_before_retract"] = True
        except Exception as exc:
            result["recovery_table_error"] = str(exc)
    elif released and near_shelf and current is not None and not result.get("recovery_released_retreat_completed"):
        try:
            _drive_line_task3(
                node, current, pose_offset(current, args.shelf_retreat, reverse=True), -1,
                args.retreat_timeout, result,
                context.get("release_left", context.get("hold_left")),
                context.get("release_right", context.get("hold_right")), args.gripper_open,
                require_hold=False, min_traveled_m=0.20, max_linear_speed=TASK3_HOLD_LINEAR_SPEED,
                key_prefix="recovery_released_retreat",
            )
            result["recovery_released_retreat_completed"] = True
        except Exception as exc:
            result["recovery_released_retreat_error"] = str(exc)
    can_retract = (
        context.get("initial_left") is not None
        and (
            released
            or context.get("hold_left") is None
            or result.get("recovery_lowered_before_retract")
            or result.get("recovery_table_error")
        )
    )
    if can_retract:
        try:
            _retract_arms(node, context, args, result)
        except Exception as exc:
            result["recovery_error"] = str(exc)
    elif holding:
        result["recovery_kept_hold"] = True
        try:
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
        except Exception as exc:
            result["recovery_keep_hold_error"] = str(exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 3 cube-top hug-to-shelf L1 placement check")
    parser.add_argument("--box-world", nargs=3, type=float, help="optional cube-top yellow center override")
    parser.add_argument("--no-allow-fixed-yellow", action="store_false", dest="allow_fixed_yellow")
    parser.set_defaults(allow_fixed_yellow=True)
    parser.add_argument("--place-world", nargs=3, type=float, default=INSTRUCTION_PLACE_WORLD.tolist())
    parser.add_argument("--place-radius", type=float, default=PLACE_RADIUS_M)
    parser.add_argument("--place-accept-radius", type=float, default=PLACE_ACCEPT_RADIUS_M)
    parser.add_argument("--place-yaw", type=float, default=PLACE_YAW)
    parser.add_argument("--grasp-yaw", type=float, default=GRASP_YAW)
    parser.add_argument("--standoff", type=float, default=TASK3_STANDOFF_M)
    parser.add_argument("--initial-clearance", type=float, default=0.02)
    parser.add_argument("--contact-step", type=float, default=0.01)
    parser.add_argument("--grasp-fwd-offset", type=float, default=TASK3_GRASP_FWD_OFFSET_M)
    parser.add_argument("--grasp-z-offset", type=float, default=TASK3_GRASP_Z_OFFSET_M)
    parser.add_argument("--approach-half", type=float, default=TASK1_APPROACH_HALF_M)
    parser.add_argument("--hold-half", type=float, default=TASK3_HOLD_HALF_M)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-max-step", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=TASK3_LIFT_HEIGHT_M)
    parser.add_argument("--table-leave-distance", type=float, default=TABLE_LEAVE_DISTANCE_M)
    parser.add_argument("--place-clearance", type=float, default=TASK3_PLACE_CLEARANCE_M)
    parser.add_argument("--release-spread", type=float, default=TASK3_RELEASE_SPREAD_M)
    parser.add_argument("--staging-back", type=float, default=STAGING_BACK_M)
    parser.add_argument("--shelf-retreat", type=float, default=SHELF_RETREAT_M)
    parser.add_argument("--joint-max-step", type=float, default=0.05)
    parser.add_argument("--spine-max-step", type=float, default=MAX_SPINE_STEP)
    parser.add_argument("--spine-tolerance", type=float, default=0.010)
    parser.add_argument("--settle-timeout", type=float, default=15.0)
    parser.add_argument("--squeeze-seconds", type=float, default=DEFAULT_SQUEEZE_SECONDS)
    parser.add_argument("--nav-timeout", type=float, default=DEFAULT_NAV_TIMEOUT_SEC)
    parser.add_argument("--yaw-timeout", type=float, default=DEFAULT_YAW_TIMEOUT_SEC)
    parser.add_argument("--table-leave-timeout", type=float, default=DEFAULT_LINE_TIMEOUT_SEC)
    parser.add_argument("--shelf-timeout", type=float, default=DEFAULT_SHELF_TIMEOUT_SEC)
    parser.add_argument("--retreat-timeout", type=float, default=DEFAULT_LINE_TIMEOUT_SEC)
    parser.add_argument("--no-detect-obstacle", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="/tmp/task3_cube_top_shelf_place_check.json")
    args = parser.parse_args(argv)
    if not 0.02 <= args.joint_max_step <= MAX_ARM_STEP or not 0.02 <= args.spine_max_step <= MAX_SPINE_STEP:
        parser.error("joint/spine max steps are outside safety bounds")
    if not 0.0 <= args.gripper_open <= 1.0:
        parser.error("gripper-open is invalid")
    if args.settle_timeout <= 0 or args.nav_timeout <= 0 or args.yaw_timeout <= 0 or args.squeeze_seconds <= 0:
        parser.error("timeout arguments are invalid")
    if not TASK3_HOLD_HALF_M <= args.hold_half <= args.approach_half <= TASK1_APPROACH_HALF_M:
        parser.error("hold-half must be within [0.115, approach-half], and approach-half <= 0.13 m")
    if not 0.06 <= args.place_accept_radius <= args.place_radius:
        parser.error("place-accept-radius must be within [0.06, place-radius]")
    if not lift_clears_cube(YELLOW_FIXED_WORLD[2], args.lift_height):
        parser.error("lift height would not clear the white cube top")
    place_world = validate_shelf_side_place(args.place_world)
    place_yaw = wrap_to_pi(args.place_yaw)
    grasp_yaw = wrap_to_pi(args.grasp_yaw)
    print(
        f"task3_cube_top_shelf_place_check starting apply={bool(args.apply)} output={args.output}",
        flush=True,
    )

    result = {
        "mode": "task3_bimanual_hug_cube_top_shelf_place_check",
        "apply": bool(args.apply),
        "formal_motion_stays_disabled": True,
        "box_contact_commanded": bool(args.apply),
        "base_motion_commanded": bool(args.apply),
        "transport_or_place_commanded": bool(args.apply),
        "gripper_open_target": args.gripper_open,
        "approach_half_m": args.approach_half,
        "hold_half_m": args.hold_half,
        "grasp_fwd_offset_m": args.grasp_fwd_offset,
        "grasp_z_offset_m": args.grasp_z_offset,
        "lift_height_m": args.lift_height,
        "table_leave_distance_m": args.table_leave_distance,
        "table_leave_min_traveled_m": TABLE_LEAVE_MIN_TRAVELED_M,
        "place_world": place_world.tolist(),
        "place_radius_m": args.place_radius,
        "place_accept_radius_m": args.place_accept_radius,
        "place_yaw_rad": place_yaw,
        "grasp_yaw_rad": grasp_yaw,
        "standoff_m": args.standoff,
        "place_clearance_m": args.place_clearance,
        "release_spread_m": args.release_spread,
        "staging_back_m": args.staging_back,
        "shelf_retreat_m": args.shelf_retreat,
        "published_control_topics": [],
        "phase": "init",
    }
    node = None
    context = {
        "initial_left": None,
        "hold_left": None,
        "hold_right": None,
        "release_left": None,
        "release_right": None,
    }
    start_base = None
    phase = "init"
    released = False
    command_issued = False
    try:
        node = Ros2MissionNode(node_name="task3_cube_top_shelf_place_check")
        node.wait_for_robot_state(timeout_sec=10.0)
        initial_left, initial_right, raw_left_gripper, raw_right_gripper = _current_arm_state_unbounded(node)
        initial_left_gripper = command_gripper_value(raw_left_gripper)
        initial_right_gripper = command_gripper_value(raw_right_gripper)
        initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        start_base = _odom_pose(node)
        context.update({
            "initial_left": initial_left,
            "initial_right": initial_right,
            "initial_left_gripper": initial_left_gripper,
            "initial_right_gripper": initial_right_gripper,
            "initial_slide": initial_slide,
        })
        result["initial_base"] = start_base.tolist()
        result["initial_slide"] = initial_slide
        carrying = already_carrying_box(initial_slide, initial_left, initial_right)
        in_shelf = float(start_base[0]) <= SHELF_ZONE_X_MAX
        resume_transport = bool(carrying and not in_shelf)
        resume_shelf = bool(carrying and in_shelf)
        result["resumed_transport"] = resume_transport
        result["resumed_shelf"] = resume_shelf
        if resume_transport:
            print("resuming held transport; skipping cube-top grasp", flush=True)
        if resume_shelf:
            print("resuming in-shelf hug; skipping grasp and table leave", flush=True)
        if args.apply and not carrying and initial_slide > 0.10:
            print(f"restoring high posture from leftover slide {initial_slide:.3f} m", flush=True)
            command_issued = True
            high_plan = solve_bimanual_hug_pose(initial_slide, initial_left, initial_right)
            high_left, high_right = _traverse_pair(
                node, initial_left, initial_right,
                high_plan["left_joint_target"], high_plan["right_joint_target"],
                args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
                initial_left_gripper, initial_right_gripper, True, result,
            )
            _traverse_spine(
                node, initial_slide, min(0.03, initial_slide), args.spine_max_step,
                args.settle_timeout, args.spine_tolerance, True, result,
            )
            initial_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            initial_left, initial_right = high_left, high_right
            initial_left_gripper = command_gripper_value(args.gripper_open)
            initial_right_gripper = command_gripper_value(args.gripper_open)
            context.update({
                "initial_left": initial_left,
                "initial_right": initial_right,
                "initial_left_gripper": initial_left_gripper,
                "initial_right_gripper": initial_right_gripper,
                "initial_slide": initial_slide,
            })
            result["initial_slide"] = initial_slide
            result["restored_high_posture"] = True
            if initial_slide > 0.10:
                raise RuntimeError(f"slide still {initial_slide:.4f} m after high-posture restore")

        phase = "detect"
        result["phase"] = phase
        keeper = _bind_capped_hold_keeper(context, result)
        located = None
        box_world = None
        if args.box_world is not None:
            box_world = snap_cube_top_center(validate_yellow_world(args.box_world))
            located = {"center_world": box_world.tolist(), "source": "cli"}
        elif carrying:
            box_world = YELLOW_FIXED_WORLD.copy()
            located = {"center_world": box_world.tolist(), "source": "resume_carry"}
        elif not args.apply:
            box_world = YELLOW_FIXED_WORLD.copy()
            located = {"center_world": box_world.tolist(), "source": "fixed_layout_dry_run"}
        else:
            current_pose = _odom_pose(node)
            if should_backup_to_observe(current_pose):
                result["observe_backup_from"] = current_pose.tolist()
                command_issued = True
                _backup_reverse(node, OBSERVE_BACKUP_M, OBSERVE_BACKUP_TIMEOUT_SEC, result)
            _look_at_table(node, result)
            last_exc = None
            try:
                located = locate_yellow(node, grasp_yaw)
            except Exception as exc:
                result["vision_error"] = str(exc)
                last_exc = exc
                located = None
            if located is None:
                for _attempt in range(MAX_OBSERVE_BACKUP_ATTEMPTS):
                    current_pose = _odom_pose(node)
                    if not should_backup_to_observe(current_pose):
                        break
                    result["observe_backup_attempts"] = int(result.get("observe_backup_attempts", 0)) + 1
                    command_issued = True
                    _backup_reverse(node, OBSERVE_BACKUP_M, OBSERVE_BACKUP_TIMEOUT_SEC, result)
                    _look_at_table(node, result)
                    try:
                        located = locate_yellow(node, grasp_yaw)
                        result["detection_after_observe_backup"] = True
                        break
                    except Exception as retry_exc:
                        result["vision_error"] = str(retry_exc)
                        last_exc = retry_exc
                        located = None
            if located is None:
                near_table = should_backup_to_observe(_odom_pose(node))
                if near_table or not args.allow_fixed_yellow:
                    raise RuntimeError(
                        f"yellow box not visible ({last_exc}); refusing the nominal cube-top "
                        "hug because the box may have moved. Reset the Server scene and retry."
                    ) from last_exc
                box_world = YELLOW_FIXED_WORLD.copy()
                located = {"center_world": box_world.tolist(), "source": "fixed_layout_fallback"}
        if box_world is None:
            box_world = snap_cube_top_center(located["center_world"])
        result["detection"] = located
        result["box_world_snapped"] = box_world.tolist()
        raw_center = located.get("center_world_raw", located.get("center_world"))
        if raw_center is not None:
            result["hug_center_y_correction_m"] = float(box_world[1] - np.asarray(raw_center, dtype=float)[1])
        print(f"phase=detect detection_source={located.get('source')}", flush=True)

        station = station_for_yellow(box_world, args.standoff, grasp_yaw)
        result["station_target"] = station.tolist()
        place_stand = place_stand_from_goal(place_world, place_yaw, np.array([0.56, 0.0, box_world[2] + args.lift_height]))
        stage_pose, insert_xy = aisle_staging_from_stand(place_stand, place_yaw, args.staging_back)
        result["place_stand_xy_nominal"] = place_stand.tolist()
        result["staging_pose_nominal"] = stage_pose.tolist()
        result["shelf_insert_xy_nominal"] = insert_xy.tolist()

        contact_slide = float(PRE_GRASP_Z0 - (box_world[2] + args.grasp_z_offset))
        if not SLIDE_LIMITS[0] <= contact_slide <= SLIDE_LIMITS[1]:
            raise RuntimeError(f"contact slide {contact_slide:.4f} is outside {SLIDE_LIMITS}")
        lift_slide = lift_slide_target(contact_slide, args.lift_height)
        result["contact_slide"] = contact_slide
        result["lift_slide"] = lift_slide
        held_after_lift = apply_slide_keep_hold(box_world, contact_slide, lift_slide)
        clearance_slide = slide_for_held_z(lift_slide, held_after_lift[2], place_world[2] + args.place_clearance)
        place_slide = slide_for_held_z(clearance_slide, apply_slide_keep_hold(held_after_lift, lift_slide, clearance_slide)[2], place_world[2])
        result["place_clearance_slide"] = clearance_slide
        result["place_slide"] = place_slide
        if max(clearance_slide, place_slide) > SLIDE_LIMITS[1] - 0.02:
            result["low_shelf_slide_margin_warning"] = True

        if not args.apply:
            result["status"] = "dry_run"
            result["box_contact_commanded"] = False
            result["base_motion_commanded"] = False
            result["transport_or_place_commanded"] = False
            print(
                "task3 dry-run ok (this is not a live pass); "
                f"contact_slide={contact_slide:.3f} lift_slide={lift_slide:.3f} "
                f"place_slide={place_slide:.3f} station={station.tolist()}",
                flush=True,
            )
            return 0

        command_issued = True
        if not carrying:
            phase = "station"
            result["phase"] = phase
            _navigate(
                node, station, 0.04, 0.03, args.nav_timeout, MAX_STATION_NAV_M,
                TASK3_HOLD_LINEAR_SPEED, TASK3_HOLD_ANGULAR_SPEED, result,
            )
            station_navigation = {
                "final_base": result.get("final_base"),
                "remaining_position_error_m": result.get("remaining_position_error_m"),
                "remaining_yaw_error_rad": result.get("remaining_yaw_error_rad"),
                "navigation_phase": result.get("navigation_phase"),
            }
            result["station_navigation"] = station_navigation
            box_base = np.asarray(_world_to_base(node, box_world), dtype=float)
            result["box_base_at_hug"] = box_base.tolist()
            raw = located.get("center_world_raw")
            if raw is not None:
                result["box_base_from_raw"] = np.asarray(_world_to_base(node, raw), dtype=float).tolist()
            if not (HUG_WINDOW_X[0] <= box_base[0] <= HUG_WINDOW_X[1] and abs(box_base[1]) <= HUG_WINDOW_ABS_Y):
                raise RuntimeError(f"yellow box is outside the hug window after station: {box_base.tolist()}")

            phase = "hug_lift"
            result["phase"] = phase
            context.update(_establish_cube_top_hold(node, args, box_base, result))
            context["contact_slide"] = result["contact_slide"]
            start_base = _odom_pose(node)
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            result["lift_completed"] = True
            _sleep_holding(
                node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
                hold_keeper=keeper,
            )

            phase = "table_leave"
            result["phase"] = phase
            table_leave_target = pose_offset(start_base, args.table_leave_distance, reverse=True)
            result["table_leave_target"] = table_leave_target.tolist()
            leave_final = _drive_line_task3(
                node, start_base, table_leave_target, -1, args.table_leave_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, min_traveled_m=TABLE_LEAVE_MIN_TRAVELED_M,
                max_linear_speed=TASK3_TABLE_LEAVE_LINEAR_SPEED, key_prefix="table_leave",
            )
            result["table_leave_final_base"] = leave_final.tolist()
            result["table_leave_completed"] = True
            if float(result.get("table_leave_traveled_m", 0.0)) < TABLE_LEAVE_MIN_TRAVELED_M:
                raise RuntimeError(
                    f"table leave traveled {result.get('table_leave_traveled_m')} m, "
                    f"need at least {TABLE_LEAVE_MIN_TRAVELED_M:.2f} m"
                )
        else:
            result["lift_completed"] = True
            current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
            try:
                context["held_center_base"] = held_center_from_palms_l1(current_slide, initial_left, initial_right)
                result["held_center_source"] = "palm_fk"
            except Exception as exc:
                result["palm_center_error"] = str(exc)
                context["held_center_base"] = np.asarray(_world_to_base(node, box_world), dtype=float)
                result["held_center_source"] = "snapped_vision"
            context["hold_left"], context["hold_right"] = local_carry_hold(initial_left, initial_right)
            result["hold_joint_targets"] = {
                "left": np.asarray(context["hold_left"]).tolist(),
                "right": np.asarray(context["hold_right"]).tolist(),
            }
            result["held_center_base_after_lift"] = context["held_center_base"].tolist()
            _command_hug(node, context["hold_left"], context["hold_right"], args.gripper_open)
            if in_shelf:
                print("held in shelf; reversing out before continuing", flush=True)
                command_issued = True
                retreat_start = _odom_pose(node)
                retreat_target = pose_offset(retreat_start, args.shelf_retreat, reverse=True)
                _drive_line_task3(
                    node, retreat_start, retreat_target, -1, args.retreat_timeout, result,
                    context["hold_left"], context["hold_right"], args.gripper_open,
                    require_hold=True, min_traveled_m=0.20, position_tolerance=0.06,
                    max_linear_speed=TASK3_SHELF_LINEAR_SPEED,
                    key_prefix="resume_held_retreat", hold_keeper=keeper,
                )
                result["resume_held_retreat_completed"] = True
                in_shelf = False

        phase = "face_west"
        result["phase"] = phase
        _face_yaw_holding(
            node, place_yaw, args.yaw_timeout, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            yaw_tolerance=PLACE_ALIGN_YAW_TOLERANCE_RAD,
            key_prefix="face_west", hold_keeper=keeper,
            max_angular_speed=TASK3_HOLD_ANGULAR_SPEED,
        )

        phase = "staging_nav"
        result["phase"] = phase
        place_stand = place_stand_from_goal(place_world, place_yaw, context["held_center_base"])
        stage_pose, insert_xy = aisle_staging_from_stand(place_stand, place_yaw, args.staging_back)
        result["place_stand_xy"] = place_stand.tolist()
        result["staging_pose"] = stage_pose.tolist()
        result["shelf_insert_xy"] = insert_xy.tolist()
        staging_final = _navigate_holding(
            node, stage_pose, args.nav_timeout, MAX_STAGING_NAV_M, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            position_tolerance=0.04, yaw_tolerance=PLACE_ALIGN_YAW_TOLERANCE_RAD,
            max_linear_speed=TASK3_HOLD_LINEAR_SPEED,
            max_angular_speed=TASK3_HOLD_ANGULAR_SPEED,
            hold_keeper=keeper,
        )
        result["staging_final_base"] = staging_final.tolist()
        result["staging_completed"] = True
        _sleep_holding(
            node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )

        phase = "locate_obstacle"
        result["phase"] = phase
        approach_hold_z = TASK3_APPROACH_Z_M
        if not args.no_detect_obstacle:
            _look_at_shelf_obstacle(node, result)
            try:
                located_obs = locate_packaging(node, place_yaw)
                derived = place_left_of_obstacle(located_obs["center_world"])
                place_world = derived
                approach_hold_z = approach_z_over_obstacle(located_obs["center_world"][2])
                result["obstacle_detection"] = located_obs
                result["place_world"] = place_world.tolist()
                result["place_world_source"] = "obstacle_left"
                result["detected_obstacle_layer_z_m"] = nearest_shelf_board_z(located_obs["center_world"][2])
            except Exception as exc:
                result["obstacle_detection_error"] = str(exc)
                derived = place_left_of_obstacle(PACKAGING_WORLD)
                place_world = derived
                result["place_world"] = place_world.tolist()
                result["place_world_source"] = "layout_packaging"
                result["obstacle_detection"] = {
                    "center_world": PACKAGING_WORLD.tolist(),
                    "source": "layout",
                }
                result["detected_obstacle_layer_z_m"] = nearest_shelf_board_z(PACKAGING_WORLD[2])
        else:
            result["place_world_source"] = "instruction"
        result["approach_hold_z_m"] = float(approach_hold_z)

        phase = "shelf_lower"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        clearance_slide = slide_for_held_z(
            current_slide, context["held_center_base"][2], approach_hold_z,
        )
        result["place_clearance_slide"] = clearance_slide
        result["approach_hold_z_m"] = float(approach_hold_z)
        result["approach_clears_packaging"] = approach_clears_packaging(approach_hold_z)
        result["cubby_fits_l1"] = cubby_fits_l1(approach_hold_z)
        obstacle_z = float(PACKAGING_WORLD[2])
        located_obs = result.get("obstacle_detection")
        if located_obs is not None:
            obstacle_z = float(located_obs["center_world"][2])
        result["place_is_l1_layer"] = place_is_l1_layer(place_world[2], packaging_z=obstacle_z)
        _traverse_spine_holding(
            node, current_slide, clearance_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
            context["hold_left"], context["hold_right"], args.gripper_open, result,
            hold_keeper=keeper,
        )
        context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, clearance_slide)
        try:
            left_now, right_now, _, _ = _current_arm_state_unbounded(node)
            context["held_center_base"] = held_center_from_palms_l1(
                float(node.sensors.joint_vector(["slide_joint"])[0]), left_now, right_now,
            )
            result["held_center_source_at_clearance"] = "palm_fk"
        except Exception as exc:
            result["clearance_palm_center_error"] = str(exc)
        result["held_center_base_at_clearance"] = context["held_center_base"].tolist()
        result["shelf_lower_completed"] = True

        phase = "shelf_approach"
        result["phase"] = phase
        approach_start = _odom_pose(node)
        place_stand_now = place_stand_from_goal(place_world, place_yaw, context["held_center_base"])
        insert_plan = south_then_west_insert_plan(approach_start[:2], place_stand_now, place_yaw=place_yaw)
        south_now = insert_plan["south_xy"]
        result["place_stand_xy"] = place_stand_now.tolist()
        result["shelf_south_xy"] = south_now.tolist()
        result["shelf_approach_south_then_west"] = True
        result["shelf_approach_south_bearing_rad"] = insert_plan["south_bearing"]
        if insert_plan["needs_south_shift"]:
            _face_yaw_holding(
                node, insert_plan["south_bearing"], args.yaw_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                yaw_tolerance=0.08, key_prefix="shelf_approach_face_south", hold_keeper=keeper,
                max_angular_speed=TASK3_HOLD_ANGULAR_SPEED,
            )
            south_start = _odom_pose(node)
            south_pose = np.array([south_now[0], south_now[1], south_start[2]], dtype=float)
            try:
                _drive_line_task3(
                    node, south_start, south_pose, 1, args.shelf_timeout, result,
                    context["hold_left"], context["hold_right"], args.gripper_open,
                    require_hold=True, position_tolerance=0.03, max_linear_speed=TASK3_SHELF_LINEAR_SPEED,
                    key_prefix="shelf_approach_south",
                    hold_keeper=keeper,
                )
            except TimeoutError as exc:
                approach_final = _odom_pose(node)
                result["shelf_approach_south_timeout_base"] = approach_final.tolist()
                raise TimeoutError(f"shelf_approach_south timed out; final={approach_final.tolist()}") from exc
        _face_yaw_holding(
            node, insert_plan["west_yaw"], args.yaw_timeout, result,
            context["hold_left"], context["hold_right"], args.gripper_open,
            yaw_tolerance=PLACE_INSERT_YAW_TOLERANCE_RAD,
            key_prefix="shelf_approach_square_west", hold_keeper=keeper,
            max_angular_speed=TASK3_HOLD_ANGULAR_SPEED,
        )
        west_start = _odom_pose(node)
        west_pose = np.array([place_stand_now[0], place_stand_now[1], insert_plan["west_yaw"]], dtype=float)
        if float(np.linalg.norm(west_start[:2] - place_stand_now)) > 0.03:
            try:
                _drive_line_task3(
                    node, west_start, west_pose, 1, args.shelf_timeout, result,
                    context["hold_left"], context["hold_right"], args.gripper_open,
                    require_hold=True, position_tolerance=0.03, max_linear_speed=TASK3_SHELF_LINEAR_SPEED,
                    key_prefix="shelf_approach_west",
                    held_center_base=context["held_center_base"],
                    place_world=place_world,
                    place_radius=args.place_accept_radius,
                    hold_keeper=keeper,
                )
            except TimeoutError as exc:
                approach_final = _odom_pose(node)
                inside = box_inside_place_radius(
                    approach_final, context["held_center_base"], place_world, args.place_accept_radius,
                )
                result["shelf_approach_west_timeout_base"] = approach_final.tolist()
                result["shelf_approach_timeout_estimated_place_world"] = inside["held_world"]
                result["shelf_approach_timeout_xy_error_m"] = inside["xy_error_m"]
                depth = shelf_inward_ok(inside["held_world"], place_world)
                result["shelf_approach_timeout_outward_m"] = depth["outward_m"]
                if not inside["within_radius"] or not depth["deep_enough"]:
                    raise
                result["shelf_approach_accepted_inside_radius"] = True
                result["shelf_approach_timeout_error"] = str(exc)
        approach_final = _odom_pose(node)
        result["place_final_base"] = approach_final.tolist()
        result["shelf_approach_completed"] = True
        _sleep_holding(
            node, 0.4, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )

        phase = "place_lower"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        place_slide = slide_for_held_z(current_slide, context["held_center_base"][2], place_world[2])
        result["place_slide"] = place_slide
        result["place_layer_z_m"] = float(place_world[2])
        result["packaging_center_z_m"] = float(PACKAGING_WORLD[2])
        _traverse_spine_holding(
            node, current_slide, place_slide, args.spine_max_step, args.settle_timeout, args.spine_tolerance,
            context["hold_left"], context["hold_right"], args.gripper_open, result,
            hold_keeper=keeper,
        )
        context["held_center_base"] = apply_slide_keep_hold(context["held_center_base"], current_slide, place_slide)
        estimated = held_center_world(_odom_pose(node), context["held_center_base"])
        error = task3_placement_error(estimated, place_world, args.place_radius)
        result["estimated_place_world"] = estimated.tolist()
        result["place_xy_error_m"] = error["xy_error_m"]
        result["place_z_error_m"] = error["z_error_m"]
        result["place_within_radius"] = error["within_radius"]
        result["place_lower_completed"] = True
        _sleep_holding(
            node, 1.0, context["hold_left"], context["hold_right"], args.gripper_open,
            hold_keeper=keeper,
        )
        release_ready = box_inside_place_radius(
            _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
        )
        result["pre_release_xy_error_m"] = release_ready["xy_error_m"]
        release_depth = shelf_inward_ok(release_ready["held_world"], place_world)
        result["pre_release_outward_m"] = release_depth["outward_m"]
        if not release_ready["within_radius"] or not release_depth["deep_enough"]:
            creep_start = _odom_pose(node)
            creep_target = np.array(
                [place_stand_now[0], place_stand_now[1], insert_plan["west_yaw"]], dtype=float,
            )
            result["l1_creep_target"] = creep_target.tolist()
            _drive_line_task3(
                node, creep_start, creep_target, 1, args.shelf_timeout, result,
                context["hold_left"], context["hold_right"], args.gripper_open,
                require_hold=True, position_tolerance=0.03, max_linear_speed=TASK3_SHELF_LINEAR_SPEED,
                key_prefix="l1_creep",
                held_center_base=context["held_center_base"],
                place_world=place_world,
                place_radius=args.place_accept_radius,
                hold_keeper=keeper,
            )
            release_ready = box_inside_place_radius(
                _odom_pose(node), context["held_center_base"], place_world, args.place_accept_radius,
            )
            result["pre_release_xy_error_m"] = release_ready["xy_error_m"]
            release_depth = shelf_inward_ok(release_ready["held_world"], place_world)
            result["pre_release_outward_m"] = release_depth["outward_m"]
            estimated = held_center_world(_odom_pose(node), context["held_center_base"])
            error = task3_placement_error(estimated, place_world, args.place_radius)
            result["estimated_place_world"] = estimated.tolist()
            result["place_xy_error_m"] = error["xy_error_m"]
            result["place_z_error_m"] = error["z_error_m"]
            result["place_within_radius"] = error["within_radius"]
        if not release_ready["within_radius"] or not release_depth["deep_enough"]:
            raise RuntimeError(
                f"refusing to release on the shelf lip; xy error {release_ready['xy_error_m']:.3f} m, "
                f"outward {release_depth['outward_m']:.3f} m"
            )
        if not error["within_radius"]:
            raise RuntimeError(
                f"estimated place xy error {error['xy_error_m']:.4f} m exceeds radius {args.place_radius:.2f} m"
            )

        phase = "release"
        result["phase"] = phase
        current_slide = float(node.sensors.joint_vector(["slide_joint"])[0])
        release_left, release_right, release_mode, release_plan = l1_release_joints(
            current_slide, context["hold_left"], context["hold_right"], args.release_spread,
        )
        result["release_mode"] = release_mode
        if release_plan is not None:
            result["release_plan"] = release_plan
        _traverse_pair(
            node, context["hold_left"], context["hold_right"], release_left, release_right,
            args.joint_max_step, args.settle_timeout, RETRACTION_JOINT_TOLERANCE_RAD,
            args.gripper_open, args.gripper_open, True, result,
        )
        context["release_left"] = release_left
        context["release_right"] = release_right
        _sleep_holding(node, 0.8, release_left, release_right, args.gripper_open, require_hold=False)
        released = True
        result["released"] = True

        phase = "shelf_retreat"
        result["phase"] = phase
        retreat_start = _odom_pose(node)
        retreat_target = pose_offset(retreat_start, args.shelf_retreat, reverse=True)
        result["shelf_retreat_target"] = retreat_target.tolist()
        retreat_final = _drive_line_task3(
            node, retreat_start, retreat_target, -1, args.retreat_timeout, result,
            release_left, release_right, args.gripper_open,
            require_hold=False, min_traveled_m=max(0.20, args.shelf_retreat - 0.08),
            position_tolerance=0.06,
            max_linear_speed=TASK3_HOLD_LINEAR_SPEED, key_prefix="shelf_retreat",
        )
        result["shelf_retreat_final_base"] = retreat_final.tolist()
        result["shelf_retreat_completed"] = True

        phase = "retract"
        result["phase"] = phase
        _retract_arms(node, context, args, result)
        result["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["failure_phase"] = phase
        print(f"task3_cube_top_shelf_place_check failed in {phase}: {exc}", flush=True)
        if command_issued and node is not None:
            try:
                _recover(node, context, args, result, start_base, phase, released, place_world)
            except Exception as recover_exc:
                result["recovery_fatal_error"] = str(recover_exc)
        return 2
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            print(f"could not write report {args.output}: {exc}", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        if node is not None:
            try:
                node.controller.stop_base()
            except Exception:
                pass
            try:
                node.close(stop_robot=False)
            except Exception:
                try:
                    node.destroy_node()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
