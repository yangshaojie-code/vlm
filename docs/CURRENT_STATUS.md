# 文旅搬运项目当前状态

更新时间：2026-08-11  
项目目录：`D:\taozhanbei\vlm_pipeline`

这是最新交接入口。旧的 `AGENT_HANDOFF.md` 和 `vlm_pipeline/ROS2_INTEGRATION_LOG.md` 保留历史过程；若内容冲突，以本文件和 `vlm_pipeline/outputs` 中的真实采集结果为准。

## 1. 当前结论

正式 ROS 2 协议、固定布局任务、RGB-D 格式、控制话题、场景尺寸、裁判阈值、官方运动学源码和机器人 MJCF 已经取得。已确认的正式参数集中在 `vlm_pipeline/COMPETITION_PARAMETERS.md`。

当前仍不能运行真实抓放，但阻塞项已收敛为动作联调：

1. `motion_planning.py` 已使用项目内的官方 NumPy `MMK2Kdl/ArmKdl`，不再引用不存在的 `discoverse.robots.MMK2IK`；单臂、双臂 FK→IK→FK 和关节限位测试已通过。
2. `head_camera_kinematics.py` 已根据 MJCF 和实时 slide/head JointState 注入 `base_link <- head_camera`；静态相机矩阵仅保留调试覆盖。仍须在真实固定布局中用 RGB-D 投影验证坐标轴和绝对误差。
3. `ros_mission_executor.py` 仍是安全禁用的单臂闭爪基线，尚未完成并验证正式的双臂 hug、搬离、放置、回结束区状态机；不得绕过 `motion_ready=False` 发送正式动作。

## 2. 正式比赛约束

- 单局 600 秒，依次完成 3 项任务。
- 每项最多 3 次尝试，失败后不得重置 Server、机器人、物品或随机种子。
- 失败后只能基于当前物理状态重新观察、导航、抓取和放置。
- 最终以 `/referee/score` 和 Server 的 `referee_results_<timestamp>.json` 为准。
- 正式 Client 不得依赖在线 API、在线下载模型或安装依赖。

## 3. 已确认的固定布局任务

真实 `/material/instruction` 已保存于 `vlm_pipeline/outputs/ros2_probe.json`：

| 任务 | 目标 | 放置类型 | place_world | radius |
| --- | --- | --- | --- | --- |
| 1 | `box_pink` / pink | `shelf_point` | `[-2.68, 0.778, 1.156]` | `0.24` |
| 2 | `box_brown` / brown | `table_point` | `[-1.00, 2.20, 0.834]` | `0.28` |
| 3 | `box_yellow` / yellow | `shelf_prop_side` | `[-2.68, 0.54, 0.498]` | `0.24` |

任务三额外字段：

```text
ref_prop=packaging_box
ref_prop_body=prop_packaging_box
direction=left
```

`/referee/gameinfo` 的真实格式：

```text
t=28.1s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-
```

当前 `mission_protocol.py` 已能解析以上格式和三任务 JSON。

## 4. ROS 2 与传感器事实

Server 节点：`/MMK2_mujoco_node`。

关键发布：instruction、referee、头部 RGB-D、左右腕 RGB、CameraInfo、JointState、Odometry、`/tf`。关键控制订阅：`/cmd_vel`、spine、head、left arm、right arm。

实测头部相机：

```text
RGB:   640x480, rgb8,   step=1920, frame_id=head_camera
Depth: 640x480, mono16, step=1280, frame_id=head_camera, 单位毫米
fx=575.2890188083568
fy=575.2890188083566
cx=320.0
cy=240.0
```

RGB 和深度时间戳同步。CameraInfo 的 `frame_id` 和畸变模型为空，但 Image 的 frame 是 `head_camera`。

JointState 共 17 项且实测均为有限值：slide、head yaw/pitch、左臂 6+夹爪、右臂 6+夹爪。

ROS TF 只发布：

```text
odom -> base_link
```

没有 `base_link -> head_camera`，相机外参必须用 MJCF 和实时 slide/head 关节自行计算。

初始机器人位姿：

```text
x=-0.70, y=0.55, yaw≈pi/2
```

## 5. 控制格式与范围

官方 README/MJCF 已确认：

```text
/cmd_vel: linear.x, angular.z
spine:    [slide_joint]
head:     [head_yaw_joint, head_pitch_joint]
left:     [joint1..joint6, gripper]
right:    [joint1..joint6, gripper]
```

MJCF 控制范围：

```text
slide:      [-0.04, 0.87], axis=[0,0,-1]
head yaw:   [-0.50, 0.50]
head pitch: [-1.18, 0.16]
gripper:    [0.0, 1.0]
```

官方 task1 固定 baseline 使用 `GRIP_OPEN=1.0`，主要通过左右臂横向内收形成 hug 抱持。官方通用单臂基类使用 `GRIP_OPEN=1.0`、`GRIP_CLOSE=0.10`。应按具体任务选择双臂抱持或单臂夹取，不能沿用当前项目的 `0.04/0.0` 未验证夹爪值。

## 6. 固定场景与裁判参数

来源：`outputs/server_reference/material_competition_layout.json` 与 `material_referee_config.json`。

```text
箱体尺寸: 0.24 x 0.16 x 0.19 m，质量 0.20 kg
桌面高度: 0.739 m
货架板面: [0.403, 0.732, 1.061, 1.366, 1.695, 2.024] m
结束区: x=[-1.15,-0.25], y=[0.10,1.00]
搬离判定: 0.20 m
稳定速度: 0.05 m/s
货架放置 z 容差: 0.16 m
掉落阈值 z: 0.30 m
碰撞记录结构: shelf, perimeter_walls
```

固定物体/箱体：pink 在桌面、yellow 在白色桌面障碍物顶部、brown 在货架 L2；白色长方体障碍物在货架 L1。

## 7. 官方运动学与动作基线

搜索结果证明官方代码使用：

```text
examples/material_sorting/mmk2_kdl.py -> MMK2Kdl
examples/material_sorting/arm_kdl.py  -> ArmKdl
```

`MMK2Kdl` 和 `ArmKdl` 为 NumPy 实现，没有 `MMK2IK`。官方原件保留在 `outputs/server_reference/`，并已复制/适配到正式源码及补齐数值测试；不要直接修改参考原件。

已确认运动学参数：

```text
SpineKdl: dx=0.033942, dz=1.406, range=[-0.04,0.87]
Spine2ArmKdl: dx=0.10704, dy=0.02283, dz=0.09475
arm joint limits:
  j1 [-3.151, 2.08]
  j2 [-2.963, 0.181]
  j3 [-0.094, 3.161]
  j4 [-3.012, 3.012]
  j5 [-1.859, 1.859]
  j6 [-3.017, 3.017]
```

官方 task1 baseline 采用：导航到观察位、视觉锁定、双臂预张开、左右同时内收、升降抬起、倒车离开、导航到货架、放置、先倒出货架再收臂、返回结束区。

## 8. 相机 MJCF 链

来源：`outputs/server_reference/mmk2_mjcf/`。

从机器人根部到相机的 MJCF 链：

```text
slide_link:      pos=[0,0,1.311], slide axis=[0,0,-1]
head_yaw_link:   pos=[0.18375,0,0.023], euler=[0,0,1.5708]
head_pitch_link: pos=[0.00099952,0.000031059,0.058], quat=[0.5,-0.5,0.5,-0.5]
camera wrapper:  pos=[0.0755,-0.1855,0], quat=[0,0.70711,0,-0.70711]
head_cam body:   pos=[-0.035,0,0], euler=[-0.33,0,0]
camera name:     head_cam
ROS Image frame: head_camera
```

当前实现已把 slide、head yaw、head pitch 实时关节值代入该链，生成 `base_link <- head_camera`；不使用永久静态矩阵替代动态外参。下一步仍需用固定布局箱体世界坐标和真实 RGB-D 投影做绝对误差与坐标轴校验。

## 9. 当前项目代码状态

已实现并保留：

- 三任务协议、gameinfo/score 解析。
- 三任务/三次机会/600 秒编排和上下文。
- ROS 任务、RGB-D、关节、里程计和 TF 订阅。
- 图像/深度解码、同步、过期和 frame 检查。
- 有界控制发布、安全停止、重复 instruction 防重置。
- 官方 `MMK2Kdl/ArmKdl` 运动学后端，含单臂/双臂、固定/搜索 slide 求解和关节限位检查。
- 基于官方 MJCF 与实时 JointState 的动态头部相机外参。
- 正式动作前预检、总动作超时和裁判结算基线。
- 本地三色 HSV/几何检测。
- 只读 `ros2_probe.py`。

最近测试：

```text
python -m unittest discover -v
Ran 43 tests
OK
```

仍待完成：

- `RosMissionExecutor` 的单臂、固定 top-down 姿态；正式任务一必须改为双臂 hug，任务二/三需要各自的已标定抓放策略。
- 固定布局下 RGB-D 反投影到已知 pink/yellow/brown 世界坐标的绝对误差与坐标轴验证。
- 受控真实 Server 动作测试：先单通道控制，再双臂接触/搬离 0.20 m、稳定放置、返回结束区，最后连续三任务与随机布局。

## 10. 已取得参考文件

目录：`vlm_pipeline/outputs/server_reference/`

```text
arm_kdl.py
mmk2_kdl.py
mmk2_fk.py
material_sorting_client_base.py
official_client_task_1_full.py
material_competition_ros2_runtime.xml
material_competition_layout.json
material_referee_config.json
mmk2_base.py
mmk2_mjcf/
kinematics_search.txt
discoverse_python_files.txt
```

`official_client_task_1.py` 是早期 `sed -n 1,360p` 截断版本；应使用 `official_client_task_1_full.py`。`ik_location.txt` 为空是历史失败记录，可忽略。

## 11. 当前任务与实施顺序

### P0：运动学后端（已完成）

1. 官方 `arm_kdl.py`、`mmk2_kdl.py` 已以项目本地模块导入，不从 `outputs` 动态 import。
2. `MMK2KdlBackend` 已支持单臂、双臂、固定/搜索 slide 高度与参考关节解排序。
3. `test_motion_planning.py` 覆盖 FK→IK→FK、双臂、关节限位和不可达目标失败关闭。
4. `formal_client --preflight-only` 会报告 IK 后端、相机 frame 和外参来源。

### P0：动态相机外参（代码与数值单测已完成）

1. 已根据 MJCF 链和 JointState 实现 `base_link <- head_camera`。
2. 已由 `Ros2MissionNode` 注入 `TransformStore`，覆盖 Server 不发布相机 TF 的缺口。
3. `test_head_camera_kinematics.py` 覆盖刚体正交性、slide 负 Z 轴和 JointState 适配；仍待真实 RGB-D 投影校验。

### P1：重写正式动作执行器

1. 参考 `official_client_task_1_full.py` 的双臂 hug、slide 和安全倒车流程。
2. 按任务位置选择桌面侧边、货架和桌面顶部抓取策略，不把固定 pink/brown/yellow 顺序写死。
3. 使用 instruction 的 `target_color/body/place_world/place_radius`。
4. 加入搬离 0.20 m、放置稳定、返回结束区和局部恢复。
5. 所有物理参数先在 `MATERIAL_RANDOMIZE=0` 下验证，再切换随机布局。

### P2：联调和工程化

1. 先只运行 `formal_client.py --preflight-only`。
2. 再执行单个有限、可停止的控制测试；不要直接运行完整三任务。
3. 完成固定任务一，再连续三任务，最后 `MATERIAL_RANDOMIZE=1`。
4. 单独处理 `MATERIAL_USE_GS=1` 的 `cuModuleGetFunction` CUDA/Triton 问题。

## 12. Client 运动学依赖检查

结果已保存到 `vlm_pipeline/outputs/client_ik_packages.txt`：

```text
mujoco: installed
scipy: installed
pinocchio: NOT INSTALLED
ikpy: NOT INSTALLED
urdf_parser_py: NOT INSTALLED
kdl_parser_py: NOT INSTALLED
```

缺失项不构成阻塞。官方 `arm_kdl.py` 只依赖 NumPy 和 Python 标准库，`mmk2_kdl.py` 只依赖 NumPy、标准库和 `ArmKdl`；不需要 Pinocchio、IKPy、URDF parser 或 KDL parser。Client 已具备移植和数值验证所需依赖。

官方 KDL 代码已作为项目本地模块接入，并已在主机完成 FK/IK 往返测试。下一步是在 Client 容器重复该数值验证，然后进行只读 preflight 和固定布局 RGB-D 投影校验；不要为此额外安装缺失的四个第三方包。

## 13. 安全边界

- 在 IK、动态相机外参和固定布局投影校验完成前，不运行不带 `--preflight-only` 的正式 Client。
- 不向控制话题发送未经限位、非有限或来源不明的数组。
- 不在失败重试时重启 Server 或清除碰撞记录。
- 不把官方 GT/perception 服务当成正式唯一感知路径，除非赛题明确允许随提交 Client 使用。
- 不修改 `outputs/server_reference` 中的官方参考原件；正式实现复制到源码模块并保留来源说明。
