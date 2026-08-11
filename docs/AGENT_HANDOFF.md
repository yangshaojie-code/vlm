# 文旅搬运项目交接说明

> **最新状态请先读 `CURRENT_STATUS.md`。** 本文件保留早期架构和环境过程，部分“尚未确认”内容已经过时；若有冲突，以 `CURRENT_STATUS.md`、`vlm_pipeline/outputs/ros2_probe.json` 和 `vlm_pipeline/outputs/server_reference/` 为准。

更新时间：2026-08-11

> 状态同步：`CURRENT_STATUS.md` 和 `vlm_pipeline/COMPETITION_PARAMETERS.md` 是当前权威入口。官方 `MMK2Kdl/ArmKdl` 已接入 `motion_planning.py`，动态 `base_link <- head_camera` 已接入 `head_camera_kinematics.py`，并有 43 个离线回归测试。本文后续保留的 KDL/相机 TODO 属于历史过程；当前真正未完成的是双臂正式动作的真实 Server 联调，`motion_ready=False` 仍是安全门。

这份文档给接手项目的 agent/开发者使用。先读本文件，再读
`vlm_pipeline/README.md` 和 `vlm_pipeline/ROS2_INTEGRATION_LOG.md`，最后运行离线测试。

## 1. 项目目标

项目要适配“面向物品识别与搬运的文旅机器人关键技术研究”正式赛题。正式程序是一个 ROS 2 Client，与官方 `material_sorting:offline-server` 分离运行，在同一局 600 秒内按 Server 给出的随机布局连续完成三个任务：

1. 从桌面白色正方体侧边抓取指定颜色箱，放入货架空层。
2. 从货架抓取指定颜色箱，放回任务一开始该箱所在的桌面侧边位置。
3. 从桌面白色正方体顶部抓取指定颜色箱，放到货架白色长方体障碍物左侧。

每项任务最多 3 次尝试；失败后只能在当前物理状态上重观测、恢复和重试，不能重启 Server、机器人、物品或随机种子。最终成绩以 Server 的 `/referee/score` 和 `referee_results_<timestamp>.json` 为准。

依据文件在工作区根目录：

- `文旅搬运赛题_变更说明.pdf`
- `DG-202612_文旅搬运赛题_常见Q&A.pdf`
- `PROJECT_SUMMARY.md`

## 2. 当前代码结构

### 已实现且已测试

- `mission_protocol.py`：解析 `/material/instruction` 的三任务 JSON；颜色别名（pink/yellow/brown、中文褐色等）；解析 gameinfo/score。
- `mission_orchestrator.py`：单局任务顺序、每项最多三次尝试、600 秒截止、成功后进入下一任务、失败后原地重试、恢复记录和上下文。
- `formal_mission_runtime.py`：以 `execute_attempt(task, context, attempt)` 回调驱动编排器；异常转为失败尝试，不主动重置场景。
- `ros2_probe.py`：只读采集官方 Server 的完整指令、裁判、图像元数据、内参、关节、里程计和 TF，输出 `outputs/ros2_probe.json`。
- `ros_contract.py`：正式话题常量和 `std_msgs/msg/String` / `Int32` 消息解包。
- `ros2_mission_node.py`：订阅任务/裁判、头部 RGB-D、腕部 RGB、CameraInfo、JointState、Odometry、TF；包含启动前反馈等待和重复指令保护。
- `ros_sensor_utils.py`：图像/深度解码、RGB-D 同步与新鲜度检查、关节反馈校验、TF 图缓存。
- `ros_robot_control.py`：带限幅和安全停止的底盘、升降、头部、双臂/夹爪发布器。
- `ros_mission_executor.py`：正式动作执行基线，串联目标检测、世界/底盘坐标转换、导航、IK、抓放、返回和裁判结算。
- `formal_client.py`：单进程三任务正式入口；动作前执行传感器、关节和 TF 预检。
- `task_targets.py`：持久化任务一开始的桌面源位置和左右槽位，供任务二使用。
- `geometry_utils.py`：基础 TF/位姿矩阵和货架层高度工具。
- `color_box_detector.py`：本地 HSV/几何三色箱检测基线，不依赖在线 VLM。

在容器内执行的离线回归结果：

```text
python3 -m unittest discover -v
Ran 36 tests
OK
```

新增覆盖包括 ROS 消息解码、RGB-D 超时/frame 检查、无效关节反馈、安全停止、重复 instruction、启动预检和第三任务裁判结算。

### 尚未完成的正式能力

当前已有完整链路的代码骨架，但仍未通过真实动作联调，尚需完成：

- 获取完整 instruction JSON，确认 `place_world`、`place_radius` 等实际字段与当前解析模型一致。
- 官方 README 已确认对齐深度为 `mono16`、单位毫米；仍需获取完整相机 TF。当前 CameraInfo 的 `frame_id` 为空，必须配置 `MATERIAL_HEAD_CAMERA_FRAME` 或 `MATERIAL_CAMERA_TO_BASE`。
- 官方 README 已确认机械臂控制数组为 `[joint1..joint6, gripper]`、升降为 `[slide_joint]`、头部为 `[yaw, pitch]`；仍需验证真实 `discoverse.robots.MMK2IK` 和夹爪数值范围。
- 用真实场景校准接近距离、抓取高度、夹爪开合值和结束区位置；当前导航/抓放仅是保守基线。
- 增加双侧接触、水平搬离至少 0.20 m、物体随夹爪移动、放置稳定、掉落和碰撞反馈判定。
- 将“停止后重试”扩展为重新观测、重新导航、重新开合夹爪等实质局部恢复策略。
- 完成固定布局三任务闭环，再完成 `MATERIAL_RANDOMIZE=1` 随机布局验收。

`client_task_1.py`、`client_task_2.py` 仍依赖旧的 DISCOVERSE 内嵌 observation/Mock IK 逻辑，只保留作历史离线示例和兼容入口，不要把它们当正式 ROS 入口。

## 3. 已确认的运行环境

已加载：

```text
material_sorting:offline-client  dc9827987e78
material_sorting:offline-server  6b3c045a0ee8
```

Client 挂载方式：

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

容器内：`ROS_DISTRO=humble`，`ros2` 位于 `/opt/ros/humble/bin/ros2`。Server/Client 必须使用相同 `ROS_DOMAIN_ID=99`、host network 和 Cyclone DDS。

Server 开发联调暂用：

```text
MATERIAL_USE_GS=0
MATERIAL_RANDOMIZE=0
MATERIAL_ENABLE_RENDER=1
MATERIAL_ENABLE_SCORE=1
```

`MATERIAL_USE_GS=1` 曾触发 TorchInductor/CUDA 错误：`undefined symbol: cuModuleGetFunction`，这属于环境/驱动兼容问题，待正式运行前单独处理。

## 4. 已确认 ROS 2 接口

Server 节点：`/MMK2_mujoco_node`。

发布：

```text
/material/instruction                                  std_msgs/msg/String
/referee/taskinfo                                       std_msgs/msg/String
/referee/gameinfo                                       std_msgs/msg/String
/referee/score                                          std_msgs/msg/Int32
/head_camera/color/image_raw                            sensor_msgs/msg/Image
/head_camera/aligned_depth_to_color/image_raw            sensor_msgs/msg/Image
/head_camera/color/camera_info                          sensor_msgs/msg/CameraInfo
/head_camera/aligned_depth_to_color/camera_info          sensor_msgs/msg/CameraInfo
/left_camera/color/image_raw                            sensor_msgs/msg/Image
/right_camera/color/image_raw                           sensor_msgs/msg/Image
/left_camera/color/camera_info                          sensor_msgs/msg/CameraInfo
/right_camera/color/camera_info                         sensor_msgs/msg/CameraInfo
/joint_states                                           sensor_msgs/msg/JointState
/slamware_ros_sdk_server_node/odom                      nav_msgs/msg/Odometry
/tf                                                     tf2_msgs/msg/TFMessage
```

订阅控制：

```text
/cmd_vel                                                geometry_msgs/msg/Twist
/spine_forward_position_controller/commands              std_msgs/msg/Float64MultiArray
/head_forward_position_controller/commands               std_msgs/msg/Float64MultiArray
/left_arm_forward_position_controller/commands           std_msgs/msg/Float64MultiArray
/right_arm_forward_position_controller/commands          std_msgs/msg/Float64MultiArray
```

控制发布器已实现并有离线单元测试，但尚未在真实 Server 上发送动作指令。

## 5. 真实消息中已经看到的值

gameinfo 样例：

```text
t=31.3s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-
```

当前任务指令可见前缀包含：`task=1`、`target_body=box_pink`、`target_color=pink`。完整 instruction 因终端编码错位且长字符串被截断，不能从该输出推断全部字段；必须由 ROS 节点获取 `message.data` 后保存原文并 JSON 解析。

头部 RGB/对齐深度 CameraInfo：`640x480`，`fx=575.2890188`、`fy=575.2890188`、`cx=320`、`cy=240`；消息里的 `frame_id` 和畸变模型为空，必须继续通过实际消息和 `/tf` 确认坐标系。

图像频率约 24 Hz；RGB 曾出现约 2.9 秒停顿，正式代码必须有有限超时和重观测逻辑。

JointState 顺序为：`slide_joint`、头部 yaw/pitch、左臂 6 关节+夹爪、右臂 6 关节+夹爪。Odometry/TF 已看到 `odom -> base_link`，初始位置约 `(-0.70, 0.55, 0.0016)`。

## 6. 下一步实施顺序

### P0：先拿到完整协议证据

在 Client 订阅已启动后重启 Server，保存完整：

```bash
ros2 topic echo /material/instruction --once
ros2 topic echo /referee/taskinfo --once
ros2 topic echo /referee/gameinfo --once
ros2 topic echo /referee/score --once
ros2 topic echo /head_camera/aligned_depth_to_color/image_raw --once
ros2 topic echo /head_camera/color/camera_info --once
ros2 topic echo /tf --once
```

重点确认：instruction 的真实嵌套字段、gameinfo 是文本还是 JSON、深度 `encoding`/`step`/单位、相机 frame、货架/桌面相关 TF frame。

### P1：完成 ROS 感知与控制适配

1. 扩展 `ros2_mission_node.py`，订阅并缓存 RGB-D、CameraInfo、JointState、Odometry、TF。
2. 新增传感器快照/有限等待 API，复用 `color_box_detector.py`，用真实 CameraInfo 和 TF 做像素到 world 的转换。
3. 在 `ros_contract.py` 中补齐所有控制话题常量和消息类型。
4. 实现 cmd_vel、升降、头部、双臂和夹爪的安全发布器，所有命令带超时和停止动作。
5. 编写真实 `execute_attempt`：导航→接近→抓取→搬离 0.20 m→搬运→放置→松爪稳定→返回结束区。
6. 为抓取失败、掉落、遮挡、重新定位增加局部恢复，不重置物理场景和碰撞记录。

### P2：逐级联调

先 `MATERIAL_RANDOMIZE=0` 完成单项任务，再完成三项连续流程；之后切换 `MATERIAL_RANDOMIZE=1` 覆盖颜色、桌面左右槽位、货架层和障碍物层随机组合。最后再处理 `MATERIAL_USE_GS=1` 和正式 Docker 提交流程。

## 7. 约束与安全边界

- 正式评测不得依赖 DashScope 或任何联网下载；`grounding.py` 只能作为实验/可选辅助，成功路径要有本地检测器。
- 不使用旧固定 `TABLE_PLACE_POSITION_BASE`、固定 yaw、固定工具桶半径作为正式目标；目标坐标必须来自 Server、观测和 TF。
- 不在失败后调用 Server reset 或重建随机场景。
- 所有等待受 MuJoCo 仿真时间和 gameinfo 剩余时间限制，不能无限阻塞。
- 不要在 Client 容器内检查 Docker；Docker 命令只在宿主 WSL/Windows 终端执行。

## 8. 快速启动检查

```bash
# 宿主机：确认两个镜像和显示环境
docker images | grep material_sorting
echo "$DISPLAY"
ls -la /tmp/.X11-unix
xhost +local:docker

# Client 容器内：确认代码和 ROS
cd /workspace/baseline
python3 -m unittest discover -v
ros2 node list
ros2 topic list -t | sort
```

若 `ros2 node list` 只有 `/parameter_events` 和 `/rosout`，说明 Server 尚未运行或 ROS_DOMAIN_ID/RMW/网络参数不一致；不要先调 Client 代码。

## 9. 交接完成标准

接手者完成以下事项后才可认为正式 Client 初步可用：完整解析三任务指令；持续获得有效 RGB-D/TF；能够发布并停止全部控制通道；固定布局下连续完成三项任务；随机布局下至少跑通一局；所有失败重试均不重置场景；Server 裁判分数和结果文件可追溯。
