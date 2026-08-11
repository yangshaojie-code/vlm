# ROS 2 联调记录

> 最新开发结论和当前任务见工作区根目录 `CURRENT_STATUS.md`；本文件主要保留按时间排列的原始联调证据。

本文件记录文旅搬运赛题 Server/Client 联调结果。数值和话题名以终端实际输出为准；后续联调请在本文件末尾追加日期和结果。

## 2026-08-11：镜像与容器

已加载镜像：

```text
material_sorting:offline-client  dc9827987e78  43.7GB (virtual) / 21.7GB
material_sorting:offline-server  6b3c045a0ee8  61.9GB (virtual) / 30.8GB
```

Client 容器使用的关键参数：

```bash
docker run --rm -it \
  --gpus all --network host --ipc host \
  --name material_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /mnt/d/taozhanbei/vlm_pipeline:/workspace/baseline \
  -w /workspace/baseline \
  material_sorting:offline-client bash
```

容器内确认：

```text
pwd                 /workspace/baseline
ROS_DISTRO          humble
which ros2          /opt/ros/humble/bin/ros2
```

Server 在 `MATERIAL_USE_GS=1` 下曾因 CUDA/TorchInductor 报错：
`undefined symbol: cuModuleGetFunction`。当前开发联调使用 `MATERIAL_USE_GS=0`，并保留 `MATERIAL_RANDOMIZE=0` 进行固定布局调试。

## 2026-08-11：ROS 2 节点与话题

Server 节点已发现：

```text
/MMK2_mujoco_node
```

发布话题：

```text
/material/instruction                                      std_msgs/msg/String
/referee/taskinfo                                           std_msgs/msg/String
/referee/gameinfo                                           std_msgs/msg/String
/referee/score                                              std_msgs/msg/Int32
/head_camera/color/image_raw                                sensor_msgs/msg/Image
/head_camera/aligned_depth_to_color/image_raw                sensor_msgs/msg/Image
/head_camera/color/camera_info                              sensor_msgs/msg/CameraInfo
/head_camera/aligned_depth_to_color/camera_info              sensor_msgs/msg/CameraInfo
/left_camera/color/image_raw                                sensor_msgs/msg/Image
/right_camera/color/image_raw                               sensor_msgs/msg/Image
/left_camera/color/camera_info                              sensor_msgs/msg/CameraInfo
/right_camera/color/camera_info                             sensor_msgs/msg/CameraInfo
/joint_states                                               sensor_msgs/msg/JointState
/slamware_ros_sdk_server_node/odom                          nav_msgs/msg/Odometry
/tf                                                         tf2_msgs/msg/TFMessage
```

控制订阅话题：

```text
/cmd_vel                                                    geometry_msgs/msg/Twist
/spine_forward_position_controller/commands                  std_msgs/msg/Float64MultiArray
/head_forward_position_controller/commands                   std_msgs/msg/Float64MultiArray
/left_arm_forward_position_controller/commands               std_msgs/msg/Float64MultiArray
/right_arm_forward_position_controller/commands              std_msgs/msg/Float64MultiArray
```

抽查结果：上述控制话题均为 Server 的订阅端，Client 尚未发布控制消息；因此当前只证明接口存在，不代表机器人已执行动作。

## 2026-08-11：裁判消息样例

`/referee/gameinfo --once`：

```text
data: t=31.3s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-
```

可确认字段：剩余/已用时间显示为 `t=31.3s`，总分为 `0`，当前为第 `1/3` 项任务，三项最佳成绩为 `[0, 0, 0]`，当前尝试为 `0`，步骤为 `-`。

`/referee/score --once`：

```text
data: 0
```

`/referee/taskinfo --once` 和 `/material/instruction` 已成功发布。该终端显示为 GBK/UTF-8 编码错位（例如 `浠诲姟`），且 `ros2 topic echo` 对长字符串只显示前缀；可见内容包含 `task=1`、`target_body=box_pink`、`target_color=pink`，任务说明前缀表示抓取粉色箱并放到原白色圆柱所在的货架层。完整 JSON 必须在 Client 内读取 `std_msgs/msg/String.data` 后保存原文并用 JSON 解析确认，不能依赖此截断终端输出。

注意：`/material/instruction` QoS 为 `RELIABLE / KEEP_LAST(2) / VOLATILE`。晚启动 Client 可能错过初始指令；调试时先启动订阅，再重启 Server。

## 2026-08-11：传感器样例

头部 RGB 与对齐深度相机信息均为 `640x480`，内参：

```text
fx = 575.2890188083568
fy = 575.2890188083566
cx = 320.0
cy = 240.0
distortion_model = ''
frame_id = ''
```

当前消息中的 `frame_id` 为空、畸变模型为空，不能直接假设相机坐标系名称；正式坐标链路必须从 `/tf` 和实际图像消息再次确认。

`/joint_states --once` 的关节顺序（17 项）：

```text
slide_joint, head_yaw_joint, head_pitch_joint,
left_arm_joint1..left_arm_joint6, left_arm_eef_gripper_joint,
right_arm_joint1..right_arm_joint6, right_arm_eef_gripper_joint
```

`/slamware_ros_sdk_server_node/odom --once`：

```text
frame_id=/odom, child_frame_id=''
position=(-0.7000000, 0.5500488, 0.0015669)
orientation=(x=0.0002456, y=-0.0002455, z=0.7071066, w=0.7071069)
```

`/tf --once` 当前包含：

```text
odom -> base_link
translation=(-0.7000000, 0.5500488, 0.0015669)
```

仍需继续核对相机 frame、深度编码/单位以及是否存在静态相机 TF。

## 2026-08-11：图像频率

```text
/head_camera/color/image_raw                       约 24.0 Hz
/head_camera/aligned_depth_to_color/image_raw      约 24.0 Hz
```

RGB 统计中曾出现一次 `max=2.897s` 的间隔，后续平均值受该停顿影响降至约 16--17 Hz；深度统计稳定在约 24 Hz。正式节点应对图像超时和停顿设置有限等待与重观测逻辑。

## 2026-08-11：Python 回归测试

在 Client 容器 `/workspace/baseline` 执行：

```bash
python3 -m unittest discover -v
```

结果：

```text
Ran 20 tests in 0.433s
OK
```

已覆盖的测试文件：

```text
test_color_box_detector.py
test_control_flow.py
test_formal_mission_runtime.py
test_mission_orchestrator.py
test_mission_protocol.py
test_task_targets.py
```

这些是离线单元测试，尚未证明真实 Server 中的抓取、导航、放置或三任务评分闭环。

## 当前核对清单

- [x] Client 镜像、工作区挂载和 ROS 2 Humble 可用
- [x] Server 节点可见，正式传感器和控制话题可见
- [x] `/referee/gameinfo`、`/referee/score` 可读
- [x] RGB-D、关节、里程计和 `/tf` 可读
- [x] 20 项离线测试通过
- [ ] 在 Client 内保存并解析完整 `/material/instruction` JSON
- [ ] 解析 `/referee/gameinfo` 的所有字段并接入超时状态机
- [x] 官方 README 确认对齐深度为 `mono16`、单位毫米
- [ ] 确认 RGB 编码和相机/机器人 TF frame
- [ ] 实现控制发布器并发送有限时长的安全测试指令
- [ ] 固定布局完成单项抓放，再完成三项连续流程
- [ ] 使用 `MATERIAL_RANDOMIZE=1` 完成至少一次完整联调
- [ ] 处理 `MATERIAL_USE_GS=1` 的 CUDA/Triton 兼容问题

## 后续追加格式

每次联调请追加以下信息，避免只记录“成功/失败”：

```text
日期与布局：
Server 启动参数：
Client 启动参数：
节点/话题检查：
完整 instruction JSON：
gameinfo/score：
传感器频率与 frame：
动作与裁判结果：
错误、碰撞、掉落和恢复：
结论与下一步：
```

## 2026-08-11：代码审查与加固

本轮针对新增正式 ROS Client 做了运行时审查，修复：重复 instruction 重置任务、异常关节反馈掩盖安全停止、过期/不同 frame 的 RGB-D 被用于规划、动作总超时未覆盖全部机械臂步骤、第三任务无法仅靠任务编号变化结算，以及首次传感器回调尚未完成就消耗尝试次数。

新增正式动作前预检：里程计、17 个关节、同步 RGB-D、相机 frame 和 world-to-camera TF 完整后，才允许进入第一次尝试。当前主机回归结果：

```text
python -m unittest discover -v
Ran 36 tests in 0.483s
OK
```

所有 Python 文件已通过 `py_compile`。这些结果仍是主机离线验证，下一步必须在 Client 容器中重复测试并运行 `python3 formal_client.py` 的预检；预检失败时不得绕过，应先补齐完整 instruction、深度编码和相机 TF。
