# DG-202612 正式参数清单

更新：2026-08-11。本文只收录已由正式 PDF、真实 ROS 采集或 Server 原始配置确认的参数。固定布局数值仅用于回归测试和投影校验；正式 Client 必须以 `/material/instruction` 和实时 RGB-D 为准。

## 正式运行

```text
Server image: material_sorting:offline-server
Client image: material_sorting:offline-client
ROS_DOMAIN_ID=99
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
MATERIAL_RANDOMIZE=1
MATERIAL_ENABLE_SCORE=1
MATERIAL_ENABLE_RENDER=1
MATERIAL_USE_GS=1
```

正式运行通常不设置 `MATERIAL_SEED`。每局 600 s MuJoCo 仿真时间，三个任务连续执行；每项最多三次尝试，取单次最高分。Client 异常退出为本次正式评测 0 分，不能通过重启 Client/Server、刷新种子或恢复物品状态继续。

运行栈：Ubuntu 22.04、ROS 2 Humble、Cyclone DDS、CUDA 12.8、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、MuJoCo、3D Gaussian Splatting。正式 Client 不得联网下载代码、模型、权重或依赖。

## 任务与裁判

`/material/instruction` 是 `std_msgs/msg/String`，内容是三项任务 JSON 列表。正式可用字段：`task`、`instruction`、`target_color`、`target_body`、`place_world`、`place_type`、`place_radius`。仍必须以视觉判断目标的实时位置，不得写死颜色、槽位或货架层。

| Task | 工作 | 触碰 | 搬离 | 放置 | 返回 | 满分 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 桌面侧边箱 -> 货架空层 | 10 | 10 | 10 | 10 | 40 |
| 2 | 货架箱 -> 任务一原桌面侧边 | 20 | 10 | 20 | 10 | 60 |
| 3 | 正方体顶部箱 -> 货架长方体障碍物左侧 | 20 | 10 | 20 | 10 | 60 |

得分顺序为触碰、夹持并搬离、正确放置、安全返回。搬离要求目标同时与左右夹持链路接触，且相对开局位置有至少 `0.20 m` 的水平位移。放置前须松开并稳定；稳定速度阈值 `0.05 m/s`。掉落结算阈值为 `z < 0.30 m`。其他 Server 阈值：参照物最大距离 `0.45 m`、默认点放置半径 `0.28 m`、侧边偏移 `0.04 m`、货架放置 z 容差 `0.16 m`。货架或外围墙碰撞只丢失本次安全返回分，不清除已得触碰/搬离/放置分。

## 场景与固定布局回归值

箱体尺寸 `0.24 x 0.16 x 0.19 m`，半尺寸 `[0.12, 0.08, 0.095]`，质量 `0.20 kg`。桌面板面 `z=0.739 m`；前三层货架板面 `z=[0.403, 0.732, 1.061] m`，对应箱体中心约 `[0.508, 0.837, 1.166] m`。货架放置 `z` 容差 `0.16 m`；结束区 `x=[-1.15,-0.25]`、`y=[0.10,1.00]`。

正式随机化是离散槽位重排：三种颜色随机分配到桌面侧边、白色正方体顶部、货架；桌面箱随机在正方体左/右；货架箱随机在前三层；白色长方体障碍物位于前三层中的另一个层位；剩余层为空层。不是约 `+/-2.5 cm` 连续扰动，也不是第二/三/四层三档。

| Task | target | place_world | radius |
| --- | --- | --- | ---: |
| 1 | `box_pink` / `pink` | `[-2.68, 0.778, 1.156]` | 0.24 |
| 2 | `box_brown` / `brown` | `[-1.00, 2.20, 0.834]` | 0.28 |
| 3 | `box_yellow` / `yellow` | `[-2.68, 0.54, 0.498]` | 0.24 |

任务三还包含 `ref_prop=packaging_box`、`ref_prop_body=prop_packaging_box`、`direction=left`。

固定布局世界坐标回归值：pink `[-1.00, 2.20, 0.834]`，yellow `[-0.54, 2.30, 1.004]`，brown `[-2.63, 0.778, 0.837]`；桌面白色正方体 `[-0.54, 2.30, 0.824]`，货架白色长方体 `[-2.63, 0.778, 0.530]`。这些值不能作为随机赛题的抓取目标写死。

## ROS、相机与控制

实测头部 RGB-D：`640x480`，RGB 为 `rgb8` / `step=1920`，深度为 `mono16` / `step=1280` / 单位 mm；图像 frame 是 `head_camera`。内参：`fx=575.2890188083568`、`fy=575.2890188083566`、`cx=320.0`、`cy=240.0`。CameraInfo 的 frame_id 为空。

关键观测话题：`/material/instruction`、`/referee/taskinfo`、`/referee/gameinfo`、`/referee/score`、头部 RGB/对齐深度及其 CameraInfo、左右腕 RGB、`/joint_states`、`/slamware_ros_sdk_server_node/odom`、`/tf`。Server 不提供二维激光雷达，也不原生提供 `/material/detections`。

Server 的 `/tf` 只包含 `odom -> base_link`，所以正式感知使用 `head_camera_kinematics.py` 根据实时 `slide_joint`、`head_yaw_joint`、`head_pitch_joint` 和官方 MJCF 构建缺失的 `base_link <- head_camera`；不依赖永久静态矩阵。初始里程计约 `x=-0.70`、`y=0.55`、`yaw=pi/2`。

控制消息：`/cmd_vel` 使用 `linear.x` 和 `angular.z`；升降 `[slide_joint]`；头部 `[head_yaw_joint, head_pitch_joint]`；每个机械臂 `[joint1..joint6, gripper]`。已确认范围：

```text
slide       [-0.04, 0.87]   axis=[0,0,-1]
head yaw    [-0.50, 0.50]
head pitch  [-1.18, 0.16]
gripper     [0.0, 1.0]
arm j1      [-3.151, 2.080]
arm j2      [-2.963, 0.181]
arm j3      [-0.094, 3.161]
arm j4      [-3.012, 3.012]
arm j5      [-1.859, 1.859]
arm j6      [-3.017, 3.017]
```

官方任务一双臂 hug 基线使用 `GRIP_OPEN=1.0`，由左右臂横向内收完成抱持；通用单臂夹取参考为 `GRIP_OPEN=1.0`、`GRIP_CLOSE=0.10`。不要使用旧的 `0.04/0.0` 夹爪值。

## 来源与状态

- 规则：`文旅搬运赛题_变更说明.pdf`、`DG-202612_文旅搬运赛题_常见Q&A.pdf`
- 实测消息：`outputs/ros2_probe.json`
- Server 原始配置：`outputs/server_reference/material_competition_layout.json`、`material_referee_config.json`
- 官方动作参考：`outputs/server_reference/official_client_task_1_full.py`

`motion_planning.py` 已接入官方 NumPy `MMK2Kdl/ArmKdl`，`head_camera_kinematics.py` 已接入动态外参。正式动作仍保持禁用，直到固定布局下完成受控的双臂接触、搬离、放置、回结束区实机验证。
