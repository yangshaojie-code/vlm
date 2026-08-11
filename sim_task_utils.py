"""DISCOVERSE/MuJoCo task success checks and object randomization helpers.

Object/joint names are scene assets, so callers configure them through environment
variables until the competition MJCF is available. Missing names fail closed instead
of reporting a false success.
"""

import os

import numpy as np


def env_name(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _mujoco():
    import mujoco

    return mujoco


def body_position(sim_node, body_name: str):
    """Return a body's world position, or None when the MJCF has no such body."""
    mujoco = _mujoco()
    body_id = mujoco.mj_name2id(sim_node.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return np.asarray(sim_node.mj_data.xpos[body_id], dtype=float).copy()


def object_is_placed_near(
    sim_node,
    object_body: str,
    reference_body: str,
    direction: str = None,
    max_xy_distance: float = 0.55,
    min_direction_offset: float = 0.05,
    max_height_delta: float = 0.35,
) -> bool:
    """Check that an object is near a reference and, optionally, on its left/right.

    Direction uses the world frame's y axis: left is +y and right is -y. This
    convention can be changed once the competition scene defines a task frame.
    """
    obj = body_position(sim_node, object_body)
    ref = body_position(sim_node, reference_body)
    if obj is None or ref is None:
        return False

    delta = obj - ref
    if np.linalg.norm(delta[:2]) > max_xy_distance or abs(delta[2]) > max_height_delta:
        return False
    if direction == "left" and delta[1] < min_direction_offset:
        return False
    if direction == "right" and delta[1] > -min_direction_offset:
        return False
    return True


def randomize_free_joints(
    sim_node,
    joint_names,
    xy_range=(-0.12, 0.12),
    yaw_range=(-np.pi, np.pi),
) -> None:
    """Randomize free-joint object poses around their MJCF reset positions."""
    mujoco = _mujoco()
    rng = getattr(sim_node, "np_random", None)
    if rng is None:
        rng = np.random.default_rng()

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(sim_node.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        if sim_node.mj_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue

        qadr = sim_node.mj_model.jnt_qposadr[joint_id]
        qpos = sim_node.mj_data.qpos[qadr : qadr + 7]
        qpos[0] += rng.uniform(*xy_range)
        qpos[1] += rng.uniform(*xy_range)
        yaw = rng.uniform(*yaw_range)
        qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    mujoco.mj_forward(sim_node.mj_model, sim_node.mj_data)
