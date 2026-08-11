"""ROS 2 topic names and message decoding kept separate from rclpy code."""

from mission_protocol import GameInfo, MissionProtocolError, MissionSpec, parse_gameinfo_payload, parse_mission_payload, parse_score_payload


INSTRUCTION_TOPIC = "/material/instruction"
TASK_INFO_TOPIC = "/referee/taskinfo"
GAME_INFO_TOPIC = "/referee/gameinfo"
SCORE_TOPIC = "/referee/score"

RGB_TOPIC = "/head_camera/color/image_raw"
DEPTH_TOPIC = "/head_camera/aligned_depth_to_color/image_raw"
RGB_CAMERA_INFO_TOPIC = "/head_camera/color/camera_info"
DEPTH_CAMERA_INFO_TOPIC = "/head_camera/aligned_depth_to_color/camera_info"
LEFT_WRIST_RGB_TOPIC = "/left_camera/color/image_raw"
RIGHT_WRIST_RGB_TOPIC = "/right_camera/color/image_raw"
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/slamware_ros_sdk_server_node/odom"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

CMD_VEL_TOPIC = "/cmd_vel"
SPINE_COMMAND_TOPIC = "/spine_forward_position_controller/commands"
HEAD_COMMAND_TOPIC = "/head_forward_position_controller/commands"
LEFT_ARM_COMMAND_TOPIC = "/left_arm_forward_position_controller/commands"
RIGHT_ARM_COMMAND_TOPIC = "/right_arm_forward_position_controller/commands"

ROS_DOMAIN_ID = "99"
RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"


def parse_instruction_message(message: object) -> MissionSpec:
    return parse_mission_payload(getattr(message, "data", message))


def parse_gameinfo_message(message: object) -> GameInfo:
    return parse_gameinfo_payload(getattr(message, "data", message))


def parse_score_message(message: object):
    return parse_score_payload(getattr(message, "data", message))


def topic_contract() -> dict:
    """Return a serializable contract useful for startup diagnostics."""
    return {
        "instruction": INSTRUCTION_TOPIC,
        "taskinfo": TASK_INFO_TOPIC,
        "gameinfo": GAME_INFO_TOPIC,
        "score": SCORE_TOPIC,
        "rgb": RGB_TOPIC,
        "depth": DEPTH_TOPIC,
        "camera_info": [RGB_CAMERA_INFO_TOPIC, DEPTH_CAMERA_INFO_TOPIC],
        "wrist_rgb": [LEFT_WRIST_RGB_TOPIC, RIGHT_WRIST_RGB_TOPIC],
        "joint_states": JOINT_STATES_TOPIC,
        "odom": ODOM_TOPIC,
        "tf": TF_TOPIC,
        "tf_static": TF_STATIC_TOPIC,
        "control": {
            "cmd_vel": CMD_VEL_TOPIC,
            "spine": SPINE_COMMAND_TOPIC,
            "head": HEAD_COMMAND_TOPIC,
            "left_arm": LEFT_ARM_COMMAND_TOPIC,
            "right_arm": RIGHT_ARM_COMMAND_TOPIC,
        },
    }
