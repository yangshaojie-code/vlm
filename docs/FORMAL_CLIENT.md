# Formal ROS 2 client

The complete sensor, TF, joint, base, arm, gripper, safety, and commissioning
contract is documented in [`PHYSICAL_INTERFACE.md`](PHYSICAL_INTERFACE.md).

`formal_client.py` is the continuous three-task entry point for the official
Client container. It subscribes to the Server instruction, referee status,
head RGB-D, CameraInfo, joint state, odometry, and TF topics; it publishes
bounded commands to `/cmd_vel`, the spine/head controllers, and both arm
controllers. Each attempt is executed in place. Failures stop the robot and
return control to the orchestrator for a retry without resetting the scene.

Run inside `material_sorting:offline-client` while the Server is publishing.
The Server currently republishes the volatile instruction, but if the client
times out waiting for it, restart the Server after the client subscription is
ready:

```bash
cd /workspace/baseline
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 formal_client.py
```

Before the first motion test, collect a read-only Server snapshot:

```bash
python3 ros2_probe.py --timeout 15 --output outputs/ros2_probe.json
```

Then run the non-motion preflight:

```bash
python3 formal_client.py --preflight-only
```

Do not start `formal_client.py` without `--preflight-only` until
`outputs/ros2_probe.json` contains no missing required topics and its
instruction payload has been checked.

The official Server images use the `head_camera` image frame, while CameraInfo
has an empty frame_id and `/tf` provides only `odom -> base_link`. The client
therefore derives `base_link <- head_camera` dynamically from the official
MJCF chain and the real-time slide/head JointState values. Do not set a static
camera matrix in normal operation. `MATERIAL_CAMERA_TO_BASE` is a temporary
debug override only and is replaced after the first valid JointState callback.
`place_world` from `/material/instruction` is authoritative; task 2 falls back
to the task-1 source position only when the Server omits it.
