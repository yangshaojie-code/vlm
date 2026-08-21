# 文旅搬运：当前任务到首次程序跑分计划

更新时间：2026-08-21  
项目目录：`D:\taozhanbei\vlm_pipeline`  
适用范围：从当前 Task 1 联调状态开始，直到首次启用正式程序进行固定布局跑分。

本文是当前联调工作的执行清单。赛题事实和长期状态仍分别以
`COMPETITION_PARAMETERS.md`、`CURRENT_STATUS.md` 和 `outputs/` 中的实测报告为准。

## 1. 安全边界

- `ros_mission_executor.py` 中必须继续保持：

  ```python
  self.motion_ready = False
  self.motion_block_reason = "dual-arm formal action strategy is not calibrated"
  ```

- 校准脚本可以通过显式 `--apply` 发布有限动作，但不得绕过正式执行器安全门。
- 任一阶段的报告不是 `status=passed` 时，禁止把该报告传给下一阶段强行执行。
- 每次真实动作都必须有里程计或关节反馈闭环、有限超时、零速度停车和异常恢复。
- 发生掉落、碰撞或物体位置改变后，必须以当前物理状态恢复；正式跑分中不得重启 Server 或重置场景。

## 2. 当前状态

### 2.1 已完成并实测通过

- ROS 2 话题、任务指令、裁判信息、RGB-D、相机内参、JointState、Odometry 和 TF 已确认。
- 粉色方块 RGB-D 定位与世界坐标/底盘坐标转换已验证。
- 头部、升降柱、左右臂、左右夹爪和双臂非接触动作已验证。
- 正方向夹爪值 `1.0` 已确认是张开；Task 1 使用双臂 hug，不依赖夹爪闭合。
- 夹爪反馈略超 `[0, 1]` 只记 warning，命令裁剪到 `[0, 1]`，不中断停车或恢复。
- 新版 Task 1 抱持参数已写入校准脚本并完成真实动作：

  ```text
  approach_half = 0.130 m
  hold_half     = 0.115 m
  grasp_fwd     = +0.065 m
  grasp_z       = +0.045 m
  gripper_open  = 1.0
  ```

- **P0 站位** `task1_precontact_check.py --stage position`：`status=passed`，`navigation_phase=complete`，位置误差约 `0.025 m`，`|yaw|` 约 `0.05 rad`。近桌面看不到箱子时，同一指令最多倒退观察 3 次。最新通过报告：`outputs/task1_precontact_position_fixed.json`（2026-08-21T04:23:01Z）。
- **P1 抱持/抬离/放回** `task1_pick_lift_check.py`：可视化与报告均通过；真实接触在 `0.01 m` 间隙，不挤到 `0 m`。
- **P2 0.20 m 短搬运** `task1_transport_check.py`：2026-08-21T04:29:50Z 可视化通过，`status=passed`，`outbound_completed=true`，`return_completed=true`。接触在 `0.01 m`，抬升和倒车全程 `holding=true`。报告：`outputs/task1_transport_official_geometry.json`。
- `ros_mission_executor.py` 仍保持 `motion_ready=False`。

### 2.2 P2 通过细节与已知缺口

最新通过运行（可视化确认箱子被抱起、倒退、放回、撤臂）：

```text
contact_clearance_detected_m = 0.01
squeeze_confirmed = true
lift_completed = true
transport_traveled_m = 0.170
transport_remaining_m = 0.030
transport_max_cross_track_m ≈ 0
transport_gripper_endpoint_warning = true  （左爪反馈 1.014，未中断）
status = passed
```

联调判定：P2 脚本与可视化通过，可以进入 P3。

评分相关缺口（不阻塞 P3，但 P7 跑分前必须收紧）：底盘在剩余 `0.03 m` 处就 `complete`，实际外移约 `0.17 m`，低于赛题箱子水平位移 `≥ 0.20 m`。货架流程里倒退离开桌边时应走到满 `0.20 m`。

失败路径里已经修过、不要回退的判定：

- `0.02 m` 到位才算接近，不当接触；只在 `≤ 0.01 m` 确认夹住。
- 确认夹住要持续约 `0.8 s`；合到空位则继续往里收。
- 抬升/倒车允许略不对称抱持（一侧仍顶着、另一侧未合空）；双臂都合空才判丢失。
- 抬升一开始就算已离桌，失败先放下再撤臂。

### 2.3 当前推进点

P0–P4 已通过。**下一步：Task 3 黄箱桌顶→货架包装箱左侧。** 不要为刷分重启 Server；粉箱在 L3，棕箱已在原粉箱桌面槽。需要 Client 容器，不要启动 `formal_client.py`。

## 3. P0：修复抓取站位最终朝向收敛

状态：已通过（2026-08-21）。

### 3.1 代码任务

- 修改 `task1_precontact_check.py` 的 `final_yaw` 控制：
  - 为非零朝向误差设置有方向的最小角速度，避免低速死区。
  - 保留最大角速度限制和 `0.05 rad` 验收阈值。
  - 单独记录进入 `final_yaw` 的时间、初始误差、最小误差和最后误差。
  - 增加朝向长期无进展检测；检测到停滞时先发布零速度，再明确失败。
  - 支持从失败的 position 报告恢复最终朝向，复用其中的目标站位和检测结果，不要求近距离重新拍摄方块。
- 增加单元测试：
  - 小角度误差输出不低于最小角速度。
  - 正负误差产生正确旋转方向。
  - 进入容差后输出 `(0, 0, complete)`。
  - 恢复模式只允许读取同一 Task 1 的有限目标数据。

### 3.2 实测流程

代码完成后，优先从当前接近站位的状态恢复；不要重新走完整 `1.1 m` 路径。预计命令形式为：

```bash
cd /workspace/baseline

python3 task1_precontact_check.py \
  --stage position \
  --position-report /tmp/task1_precontact_position_new.json \
  --nav-timeout 30 \
  --output /tmp/task1_precontact_position_fixed.json \
  --apply
```

若 Server 再次重启，则不使用旧报告恢复，重新从初始位置执行完整站位：

```bash
python3 task1_precontact_check.py \
  --stage position \
  --nav-timeout 90 \
  --output /tmp/task1_precontact_position_fixed.json \
  --apply
```

### 3.3 通过条件

```text
status == passed
navigation_phase == complete
remaining_position_error_m <= 0.03
abs(remaining_yaw_error_rad) <= 0.05
final_base 存在且为有限 [x, y, yaw]
```

## 4. P1：新版抱持、抬离和放回验证

状态：已通过（2026-08-21，真实接触在 `0.01 m`）。

站位通过后，运行新版官方几何验证：

```bash
python3 task1_pick_lift_check.py \
  --position-report /tmp/task1_precontact_position_fixed.json \
  --hold-seconds 3 \
  --output /tmp/task1_pick_lift_official_geometry.json \
  --apply
```

通过条件：

- `status=passed`、`contact_detected=true`、`lift_completed=true`。
- 左右接触残差均在脚本允许范围内，且接触保持对称。
- 可视化中方块完整离开桌面约 `0.10 m`，保持 3 秒无持续下滑、明显倾斜或单侧脱离。
- 方块被放回原桌面位置，两臂撤离并回到高位初始姿态。
- 报告中参数必须为 `hold_half=0.115`、`grasp_fwd=0.065`、`grasp_z=0.045`。

未通过时只调整抓取中心、高度或抱持半宽中的一个参数，并保留每次报告；不得直接进入底盘搬运。

## 5. P2：0.20 m 带物短距离搬运闭环

状态：已通过（2026-08-21T04:29:50Z，可视化确认）。评分位移约 `0.17 m` 的缺口见 2.2。

P1 通过后运行：

```bash
python3 task1_transport_check.py \
  --position-report /tmp/task1_precontact_position_fixed.json \
  --transport-distance 0.20 \
  --transport-timeout 18 \
  --output /tmp/task1_transport_official_geometry.json \
  --apply
```

动作闭环必须是：抓取、抬离、直线倒退 `0.20 m`、返回抓取站位、放回、撤臂。

通过条件：

- `outbound_completed=true`、`return_completed=true`、`status=passed`。
- 实际外移距离达到 `0.20 m` 的允许误差范围，横向误差和朝向误差受控。
- 搬运期间双臂接触残差持续对称，方块无滑落和碰地。
- 原始夹爪反馈即使略超端点也只形成 warning，不能终止底盘或恢复流程。
- 任意失败都先发布零速度，并尽可能回到抓取站位后放回物体。

## 6. P3：Task 1 货架搬运与放置

P2 已通过。独立校准脚本 `task1_shelf_place_check.py` 已在真实 Server 上 `--apply` 通过（2026-08-21 08:11:29Z）。粉箱落在棕色箱上一层，估计箱心 `[-2.507, 0.761, 1.156]`，XY 误差 `0.174 m`。不要重启 Server。

目标参数（固定布局 instruction，放置半径以 Server 为准）：

```text
target_body = box_pink
place_type  = shelf_point
place_world = [-2.68, 0.778, 1.156]
place_radius = 0.24
place_yaw   = π（朝西）
```

官方 baseline（`outputs/server_reference/official_client_task_1_full.py`）把粉箱放到约 `[-2.64, 0.778, 1.145]`，柜内举高余量 `0.055 m`，放置后倒出 `0.32 m` 再收臂。校准脚本应对准 instruction 的 `place_world`，本地误差落在 `0.24 m` 半径内即可。

动作顺序（对应官方状态 7–16）：

1. 桌边新版双臂抱持和抬离（沿用 P2：`0.01 m` 接触、锁定抱持关节）。
2. 朝北直线倒退离开桌边，保持抱持；此段外移应达到 `0.20 m`。
3. 导航到货架放置站位正后方约 `0.50 m` 的中转位，提前转到朝西。
4. 只动升降，把箱子举到放置面上方 `0.055 m`，不在货架近区换双臂 IK。
5. 低速直线开到放置站位。只有箱心进入 `0.18 m` 内才停、落层、松手；不得用 `0.24 m` 评分半径当“已经放到货架上”。
6. 只动升降落到放置高度，双臂各外展约 `0.04 m` 松开。
7. 保持释放姿态倒退约 `0.32 m` 离开货架，再收臂；本阶段校准不必回结束区。

通过条件：方块稳定落在目标层、释放后不随手臂移动、机器人退出时不碰货架，且本地计算的放置误差满足 `place_radius=0.24 m`。任意失败先零速度；若仍抱着且还在柜口外，保持抱持不要松手。掉到地上后再重启 Server（校准阶段允许）。

## 7. P4：Task 2 与 Task 3 独立校准

### Task 2

```text
target_body = box_brown
来源 = 货架 L2 [-2.63, 0.778, 0.837]
place_type = table_point
place_world = [-1.00, 2.20, 0.834]
place_radius = 0.28
```

独立校准脚本 `task2_shelf_to_table_check.py`：**已通过**（2026-08-21 09:12:49Z），估计放置 `[-0.988, 2.192, 0.834]`，XY 误差 `0.015 m`。本局裁判 Task 2 仍为 0，因为 600 s 在重试中耗尽。

```bash
python3 task2_shelf_to_table_check.py \
  --output /workspace/baseline/outputs/task2_shelf_to_table.json \
  --apply
```

要点：L2 抱持半宽 `0.08 m`（短边 0.16 m），只抬 `0.08 m` 以免碰到上层粉箱；倒出至少 `0.20 m`；桌面放置半径 `0.28 m`。视觉失败时用固定布局棕箱坐标兜底。失败时若仍抱着，先倒出货架再保持抱持，不要在柜口举臂。

任务：校准货架取物、货架后撤、带物导航、桌面放置和撤臂。Task 2 必须使用持久化的 Task 1 原始粉色方块桌面位置语义，不能依赖场景重置。

### Task 3

```text
target_body = box_yellow
来源 = 白色方块顶部
ref_prop_body = prop_packaging_box
direction = left
place_type = shelf_prop_side
place_world = [-2.68, 0.54, 0.498]
place_radius = 0.24
```

任务：校准顶部取物、避开白色支撑物、低层货架放置，以及相对 `packaging_box` 左侧的语义检查。

每个任务都按相同顺序验收：非接触规划、接触抬离、0.20 m 搬离、完整搬运放置。禁止直接用 Task 1 参数替代 Task 2/3 的抓取姿态。

## 8. P5：三任务连续闭环与恢复

在独立任务通过后，使用校准脚本完成固定布局三任务连续执行，验证物理状态能跨任务保持：

- Task 1 的原桌面位置被持久化，供 Task 2 放置使用。
- 每个任务最多 3 次尝试，失败后不重启 Server。
- 掉落、单侧接触、导航超时和放置不稳定都能停止并形成明确报告。
- 三个任务之间不残留非零 `/cmd_vel`，双臂和升降柱处于下一任务认可的初始状态。
- 最后能够返回结束区 `x=[-1.15,-0.25]`、`y=[0.10,1.00]`。

## 9. P6：接入正式程序

只有 P0-P5 全部通过后，才把实测状态机接入 `RosMissionExecutor`：

1. 把固定颜色顺序改为使用 `/material/instruction` 的 `target_body`、`target_color`、`place_type`、`place_world` 和 `place_radius`。
2. 保留 RGB-D/TF/JointState/Odometry 启动预检和每阶段超时。
3. 保留接触、掉落、碰撞、裁判进度和恢复日志。
4. 运行完整离线测试和 `formal_client.py --preflight-only`。
5. 在真实 Server 上运行只读正式入口，确认不会因重复 instruction 重启任务。
6. 最后审查 `motion_ready` 安全门；只有所有验收证据齐全时才允许改为启用。

启用安全门前必须满足：

```text
完整离线测试全部通过
Task 1/2/3 独立真实动作报告均 passed
固定布局三任务连续闭环 passed
所有异常路径能够 stop_base
正式入口 preflight passed
没有使用在线 API 或运行时下载依赖
```

## 10. P7：首次程序跑分

到达本阶段才开始“使用自己的程序测试跑分”。首次只跑固定布局：

```text
MATERIAL_RANDOMIZE=0
MATERIAL_USE_GS=0
MATERIAL_ENABLE_RENDER=1
MATERIAL_ENABLE_SCORE=1
```

运行正式 Client 后，必须保存：

- Client 完整日志。
- `/material/instruction` 原始 JSON。
- `/referee/gameinfo`、`/referee/score` 时间线。
- 每个任务的感知、导航、接触、搬运、放置和恢复报告。
- Server 生成的 `referee_results_<timestamp>.json`。

首次固定布局获得可重复分数后，再规划 `MATERIAL_RANDOMIZE=1` 的随机布局验收；随机布局不属于本文当前完成范围。

## 11. 当前执行顺序摘要

```text
P0 站位 passed
  -> P1 新版 Task 1 抱持/抬离/放回 passed
  -> P2 0.20 m 搬运/返回/放回 passed（可视化确认；评分位移仍约 0.17 m）
  -> P3 货架放置 passed（箱心约 [-2.507, 0.761, 1.156]，误差 0.174 m）
  -> P4 Task 2 货架→桌面 passed（09:12:49Z，XY 误差 0.015 m；本局裁判未入账因 600 s 超时）
  -> 当前：Task 3 独立闭环（黄箱桌顶→货架包装箱左侧）
  -> 固定布局三任务连续闭环
  -> 接入 RosMissionExecutor 并通过 preflight
  -> 审查并启用 motion_ready
  -> 首次固定布局正式程序跑分
```
