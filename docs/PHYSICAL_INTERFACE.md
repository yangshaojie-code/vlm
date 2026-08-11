# 文旅搬运赛题物理接口文档

更新日期：2026-08-11

## 1. 文档范围

本文档描述 `material_sorting:offline-client` 中参赛程序与
`material_sorting:offline-server` 仿真机器人之间需要调用的物理接口，包括：

- 头部 RGB-D、腕部 RGB、关节状态、里程计和 TF 感知接口；
- 底盘、升降柱、头部、左右机械臂和夹爪控制接口；
- 任务指令和裁判状态接口；
- 坐标、单位、QoS、超时、停止和故障恢复要求；
- 当前仍需通过 Server 联调确认的参数。

正式运行架构为：Server 负责 MuJoCo 场景、传感器和裁判；Client 通过 ROS 2
订阅观测并发布控制命令。Client 不得重置 Server、物体或随机种子。

## 2. 运行环境

Server 和 Client 必须使用相同的 ROS 2 通信配置：

```bash
export ROS_DOMAIN_ID=99
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

两个容器均需使用：

```text
--network host --ipc host
```

正式环境为 ROS 2 Humble 和 Cyclone DDS。当前官方 Client 镜像中没有
`discoverse` Python 包，因此正式控制程序不能依赖 `discoverse.robots.MMK2IK`，
除非将其及全部离线依赖打包进参赛镜像。当前 `formal_client.py` 会在只有 Mock IK
时拒绝启动。

## 3. 接口状态标记

| 标记 | 含义 |
|---|---|
| 已确认 | 已从 Server 的节点、话题或实际消息中看到 |
| 已实现 | Client 代码已经订阅、解码或发布，但尚未完成物理闭环 |
| 待确认 | 必须通过固定布局 Server 联调后才能作为正式参数使用 |

## 4. 任务与裁判接口

| Topic | Type | 方向 | QoS/行为 | 状态 |
|---|---|---|---|---|
| `/material/instruction` | `std_msgs/msg/String` | Server -> Client | `RELIABLE`、`KEEP_LAST(2)`、`VOLATILE` | 已确认、已实现 |
| `/referee/taskinfo` | `std_msgs/msg/String` | Server -> Client | 当前任务中文描述 | 已确认、已实现 |
| `/referee/gameinfo` | `std_msgs/msg/String` | Server -> Client | 裁判进度文本 | 已确认、已实现 |
| `/referee/score` | `std_msgs/msg/Int32` | Server -> Client | 当前总分 | 已确认、已实现 |

`/material/instruction` 的正式内容是三项任务 JSON 列表。已知任务字段包括：

```text
task
instruction
target_color
target_body
place_world
place_type
place_radius
```

Client 不得写死颜色顺序、桌面左右槽位或货架层数。由于 instruction 为
`VOLATILE`，Client 的订阅器必须在 Server 发布任务前启动。

`/referee/gameinfo` 当前样例：

```text
t=31.3s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-
```

字段含义：

| 字段 | 含义 |
|---|---|
| `t` | MuJoCo 仿真已用时间，单位秒 |
| `score` | 当前总分 |
| `task` | 当前任务序号/任务总数 |
| `best` | 三项任务当前最高分 |
| `attempt` | 当前任务尝试次数 |
| `step` | 裁判当前步骤 |

## 5. 感知接口

### 5.1 头部 RGB-D

| Topic | Type | 内容 | 状态 |
|---|---|---|---|
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部彩色图像 | 已确认、已实现 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 对齐到彩色图的深度 | 已确认、已实现 |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | RGB 内参 | 已确认、已实现 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 对齐深度内参 | 已确认、已实现 |

当前已看到的图像尺寸和内参：

```text
width  = 640
height = 480
fx = 575.2890188083568
fy = 575.2890188083566
cx = 320.0
cy = 240.0
```

Client 解码支持：

| Encoding | 解释 | Client 输出 |
|---|---|---|
| `rgb8` | RGB，每通道 8 位 | `uint8 HxWx3 RGB` |
| `bgr8` | BGR，每通道 8 位 | 转换为 `uint8 HxWx3 RGB` |
| `rgba8`/`bgra8` | 四通道彩色图 | 转换为三通道 RGB |
| `mono8`/`8UC1` | 8 位灰度 | `uint8 HxW` |
| `16UC1`/`mono16` | 深度，通常为毫米 | 乘以 `0.001`，转换为米 |
| `32FC1` | 深度，浮点米 | 保持米单位 |

无效深度（非有限值、零和负值）统一转换为 `NaN`。Client 使用图像 `step`
处理行填充，不假定数据紧密排列。

RGB 和深度快照默认要求时间戳差不超过 `0.15 s`，等待超时默认为 `3 s`。
头部图像应使用 sensor-data QoS，以兼容 `BEST_EFFORT` 发布端。

### 5.2 腕部 RGB

| Topic | Type | 内容 | 状态 |
|---|---|---|---|
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕彩色图像 | 已确认、已订阅 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕彩色图像 | 已确认、已订阅 |

当前执行器只缓存腕部消息，尚未用腕部图像完成抓取前二次精定位。这是正式抓取
成功率优化项，不应替代头部 RGB-D 的全局定位。

### 5.3 关节状态

Topic：

```text
/joint_states    sensor_msgs/msg/JointState
```

已观察到 17 个关节，顺序如下：

```text
0   slide_joint
1   head_yaw_joint
2   head_pitch_joint
3   left_arm_joint1
4   left_arm_joint2
5   left_arm_joint3
6   left_arm_joint4
7   left_arm_joint5
8   left_arm_joint6
9   left_arm_eef_gripper_joint
10  right_arm_joint1
11  right_arm_joint2
12  right_arm_joint3
13  right_arm_joint4
14  right_arm_joint5
15  right_arm_joint6
16  right_arm_eef_gripper_joint
```

正式代码必须按 `JointState.name` 查找位置，不能只依赖数组下标。机械臂动作完成
判定应使用关节反馈，当前默认最大误差为 `0.06 rad`，连续三帧满足才认为到位。

### 5.4 里程计

Topic：

```text
/slamware_ros_sdk_server_node/odom    nav_msgs/msg/Odometry
```

当前已观察到：

```text
frame_id = /odom
initial position ~= (-0.70, 0.55, 0.0016)
```

Client 使用位置 `x/y` 和四元数计算底盘 yaw。`place_world` 当前默认按 `odom`
坐标解释，可通过以下变量修改：

```bash
export MATERIAL_WORLD_FRAME=odom
```

### 5.5 TF

| Topic | Type | QoS | 状态 |
|---|---|---|---|
| `/tf` | `tf2_msgs/msg/TFMessage` | sensor-data | 已确认、已实现 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `RELIABLE` + `TRANSIENT_LOCAL` | 已实现、待确认 Server 是否发布 |

当前只确认了：

```text
odom -> base_link
```

头部图像和 CameraInfo 的 `header.frame_id` 当前样例为空。因此像素转三维点后，
还缺少“相机坐标 -> base_link/odom”的外参。正式运行前必须满足以下任一条件：

1. `/tf` 或 `/tf_static` 提供相机 frame 到 `base_link` 的完整链；
2. 从 Server 场景/MJCF 获取相机外参，并通过环境变量传入 4x4 矩阵。

配置示例：

```bash
export MATERIAL_HEAD_CAMERA_FRAME=head_camera_color_optical_frame
export MATERIAL_CAMERA_TO_BASE='[[r00,r01,r02,tx],[r10,r11,r12,ty],[r20,r21,r22,tz],[0,0,0,1]]'
```

`MATERIAL_CAMERA_TO_BASE` 表示把相机坐标中的点变换到 `base_link` 的矩阵，平移
单位为米。缺少相机 frame 或 TF 链时，Client 必须停止本次尝试，不得使用固定
占位坐标控制机械臂。

## 6. 控制接口

### 6.1 底盘

| Topic | Type | 方向 | 状态 |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Client -> Server | 已确认、已实现，待有限运动验证 |

当前只使用：

```text
linear.x   前后速度，单位 m/s
angular.z  yaw 角速度，单位 rad/s
```

Client 安全限幅：

```text
abs(linear.x)  <= 0.35 m/s
abs(angular.z) <= 0.65 rad/s
```

每段定时底盘命令最长 `15 s`。正常结束、超时和异常分支都必须发布零速度：

```yaml
linear:
  x: 0.0
angular:
  z: 0.0
```

最小安全测试命令：

```bash
timeout 1s ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.03, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

测试前必须使用固定布局、保证机器人周围无障碍，并准备立即发送停止命令。

### 6.2 升降柱

| Topic | Type | 载荷 | 状态 |
|---|---|---|---|
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[slide_position]` | 话题已确认、载荷待物理验证 |

预期数据长度为 1，单位为米。Client 当前拒绝非有限值和绝对值大于 `1.0` 的
命令。实际关节上下限必须从 Server 控制器配置或有限动作测试中确认。

### 6.3 头部

| Topic | Type | 载荷 | 状态 |
|---|---|---|---|
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[yaw, pitch]` | 话题已确认、载荷待物理验证 |

预期数据长度为 2，单位为弧度。发送前应先读取
`head_yaw_joint/head_pitch_joint`，第一次测试只发布当前反馈值，确认不会跳变。

### 6.4 左右机械臂和夹爪

| Topic | Type | 预期载荷 | 状态 |
|---|---|---|---|
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 左臂 6 关节 + 左夹爪 | 话题已确认，数组定义待物理验证 |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 右臂 6 关节 + 右夹爪 | 话题已确认，数组定义待物理验证 |

当前 Client 按以下顺序发送 7 个浮点数：

```text
[joint1, joint2, joint3, joint4, joint5, joint6, gripper]
```

机械臂关节单位预期为弧度。当前夹爪默认值：

```text
open   = 0.04
closed = 0.00
```

上述数组长度、顺序和夹爪范围尚未通过物理命令验证。验证时必须：

1. 读取 `/joint_states` 当前值；
2. 将完全相同的 7 个当前值发布回对应控制器；
3. 确认机器人无跳变；
4. 只对单个关节增加极小增量；
5. 立即恢复原值；
6. 最后才验证夹爪开合。

不得用任意零数组作为第一次机械臂测试命令，零位可能导致机械臂突然运动。

## 7. Client Python 调用接口

### 7.1 创建节点

```python
from ros2_mission_node import Ros2MissionNode

node = Ros2MissionNode()
try:
    mission = node.wait_for_mission(timeout_sec=20.0)
finally:
    node.close()
```

`close()` 会先停止底盘，并在已有反馈时向升降柱、头部和双臂发布当前位置保持。

### 7.2 获取同步 RGB-D

```python
snapshot = node.wait_for_snapshot(timeout_sec=3.0, max_skew=0.15)

rgb = snapshot.rgb                 # RGB uint8, HxWx3
depth_m = snapshot.depth_m         # float32, HxW, 单位米
intrinsics = snapshot.intrinsics
camera_frame = snapshot.camera_frame
```

像素转相机坐标：

```python
point_camera = intrinsics.project_pixel(u, v, depth_m[v, u])
```

相机转世界坐标：

```python
from geometry_utils import transform_point

camera_to_world = node.transforms.lookup("odom", camera_frame)
point_world = transform_point(camera_to_world, point_camera)
```

### 7.3 发布底盘命令

```python
node.controller.publish_velocity(linear_x=0.05, angular_z=0.0)
node.controller.stop_base()
```

有限时长命令：

```python
node.controller.drive_for(
    linear_x=0.05,
    angular_z=0.0,
    duration=1.0,
    spin_once=node.spin_once,
)
```

`drive_for()` 使用 `finally` 停止底盘。

### 7.4 发布关节命令

```python
node.controller.command_spine(position)
node.controller.command_head([yaw, pitch])
node.controller.command_arm("l", left_six_joints, left_gripper)
node.controller.command_arm("r", right_six_joints, right_gripper)
```

异常或退出：

```python
node.controller.stop_all()
```

## 8. 推荐物理动作顺序

一次抓放尝试应按以下顺序执行：

```text
等待完整 instruction
-> 获取同步 RGB-D/CameraInfo/TF/JointState/Odometry
-> 检测目标颜色并计算世界坐标
-> 记录任务一箱子的原始桌面坐标
-> 底盘接近目标并停止
-> 重新观测目标
-> 机械臂到抓取上方，夹爪打开
-> 下降并接触目标
-> 合拢夹爪
-> 抬升
-> 确认水平搬离至少 0.20 m
-> 导航到 Server 的 place_world
-> 下降
-> 松开夹爪并等待物体稳定
-> 机械臂撤离
-> 返回开局结束区
-> 等待 gameinfo/score 结算
```

任务失败后只允许在当前物理状态上重新感知、恢复和重试，不得重启 Server 或
恢复随机布局。

## 9. 安全要求

1. 所有等待必须有超时，不允许无限阻塞。
2. 所有速度和关节命令必须是有限数值并经过限幅。
3. 任何异常都必须先停止底盘，再保持当前关节位置。
4. 机械臂第一次联调不得发送零数组或大幅目标变化。
5. 目标、放置点和相机外参不得使用历史占位值。
6. 导航过程中头部中央深度小于 `0.28 m` 时停止前进。
7. 物体掉落、抓取失败或 TF 丢失时进行局部恢复，不重置场景。
8. Server 的 MuJoCo 时间达到 `600 s` 后不得开始新尝试。
9. 正式成功判定以 `/referee/score` 和 Server 结果 JSON 为准。

## 10. 联调命令

Server 启动后，在 Client 容器中执行：

```bash
ros2 node list
ros2 topic list -t | sort
ros2 topic info -v /cmd_vel
ros2 topic info -v /left_arm_forward_position_controller/commands
ros2 topic info -v /right_arm_forward_position_controller/commands
ros2 topic echo /joint_states --once
ros2 topic echo /slamware_ros_sdk_server_node/odom --once
ros2 topic echo /tf --once
ros2 topic echo /tf_static --qos-durability transient_local --once
```

在 Server 启动前挂载 instruction 监听：

```bash
mkdir -p outputs/ros_probe
ros2 topic echo /material/instruction --once > outputs/ros_probe/instruction.txt &
```

采集图像元数据时优先使用 `--field`，避免把整帧像素打印到终端：

```bash
ros2 topic echo /head_camera/color/image_raw --once --field header
ros2 topic echo /head_camera/color/image_raw --once --field encoding
ros2 topic echo /head_camera/color/image_raw --once --field step
ros2 topic echo /head_camera/aligned_depth_to_color/image_raw --once --field header
ros2 topic echo /head_camera/aligned_depth_to_color/image_raw --once --field encoding
ros2 topic echo /head_camera/aligned_depth_to_color/image_raw --once --field step
```

## 11. 正式运行前待确认项

以下项目未确认前，不应执行完整自动抓放：

| 项目 | 获取方式 | 当前影响 |
|---|---|---|
| 完整 instruction JSON | Client 先订阅，再启动 Server | 确认三任务字段和坐标语义 |
| 深度 encoding/单位 | 读取深度 Image 元数据和中心像素 | 确认毫米或米 |
| 相机 frame | `/tf`、`/tf_static` 或 MJCF | 决定像素点能否转换到世界坐标 |
| 相机到 `base_link` 外参 | TF 或 MJCF `pos/quat` | 缺失时禁止机械臂动作 |
| `place_world` 坐标系 | 与 odom/场景坐标对比 | 决定放置点和导航目标 |
| 双臂命令数组长度和顺序 | 当前值回发、小增量测试 | 防止关节跳变 |
| 夹爪开/闭范围和方向 | 小幅测试并观察 JointState | 决定能否正确夹持 |
| 升降柱/头部实际上下限 | 控制器参数或有限测试 | 防止越界 |
| 可用 IK 实现 | 检查 Client 镜像和官方示例 | 当前 `discoverse` 不存在，是正式执行阻塞项 |
| 结束区位置和判定范围 | Server 文档/裁判消息 | 决定安全返回得分 |

## 12. 分阶段验收

### 阶段 A：只读感知

- 能看到 `/MMK2_mujoco_node`；
- 能收到完整 instruction、RGB-D、CameraInfo、JointState、Odometry 和 TF；
- 深度单位和相机外参已确认；
- 不发布任何非零控制命令。

### 阶段 B：安全控制

- 发布零速度；
- 底盘以 `0.03 m/s` 移动不超过 `1 s` 并可靠停止；
- 将关节当前位置原样发布回控制器且无跳变；
- 单关节极小增量和夹爪小幅开合正常。

### 阶段 C：固定布局

- `MATERIAL_RANDOMIZE=0`；
- 完成单次定位、抓取、水平搬离、放置和返回；
- Server 分数正确增长；
- 再连续完成三项任务。

### 阶段 D：正式随机布局

- `MATERIAL_RANDOMIZE=1`；
- 不写死颜色、桌面左右位置和货架层；
- 至少完成一局三任务并保存裁判结果；
- 最后处理 `MATERIAL_USE_GS=1` 的 CUDA/TorchInductor 兼容问题。
