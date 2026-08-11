# DG-202612 文旅机器人任务:ROS 2 三任务 Client 与视觉控制

## 正式赛题适配状态

> 正式 ROS Client 的权威参数与验收状态见 `COMPETITION_PARAMETERS.md` 和
> `CURRENT_STATUS.md`。本文后续的 DashScope/DISCOVERSE、工具桶和静态相机内容只属于
> 早期离线演示链路，不能用于 DG-202612 正式评测。

当前正式要求是在同一局 600 秒内连续处理三个任务，而不是分别运行两个
DISCOVERSE Python 示例。正式 Server/Client 通过 ROS 2 通信，任务指令来自
`/material/instruction`，裁判状态来自 `/referee/taskinfo`、`/referee/gameinfo`
和 `/referee/score`。本仓库已经先实现不依赖 ROS/Docker 的协议解析、三任务编排、
局内三次尝试、局部恢复记录、货架层几何和颜色箱离线基线检测。

新增模块：

- `mission_protocol.py`：正式指令和裁判消息模型。
- `mission_orchestrator.py`：单局三任务状态机，不重置场景或随机种子。
- `formal_mission_runtime.py`：将三任务编排连接到可替换的动作执行回调。
- `ros_contract.py`：正式 ROS 2 话题契约和消息解析入口。
- `ros2_mission_node.py`：任务、裁判、RGB-D、关节、里程计和 TF 订阅，以及启动预检。
- `ros_sensor_utils.py` / `ros_robot_control.py`：ROS 消息解码、同步传感器缓存和有界控制发布。
- `ros_mission_executor.py` / `formal_client.py`：真实三任务动作基线与正式常驻入口。
- `task_targets.py` / `geometry_utils.py`：任务二源位置持久化和世界/机器人坐标工具。
- `color_box_detector.py`：不依赖在线 API 的颜色箱基线检测。

当前已在官方容器确认 ROS 2 话题类型、完整 instruction JSON、深度编码、控制数组语义和物理裁判参数；Server 只发布 `odom -> base_link`，动态相机外参已由 MJCF/JointState 实现。运行入口与必需环境变量见 `FORMAL_CLIENT.md`，正式参数清单见 `COMPETITION_PARAMETERS.md`，联调证据见 `ROS2_INTEGRATION_LOG.md`。

面向"物品识别与搬运的文旅机器人"赛题,实现完整的五段式流水线:

1. **指令解析** (`task_parser.py`):自然语言指令 → 结构化任务 JSON
2. **目标定位** (`grounding.py`):图片 + 目标描述 → 像素边界框和中心点
3. **抓取点生成** (`grasp_pose.py`):像素中心 + 深度 → base_link 坐标系三维坐标 → 抓取/放置位姿
4. **运动规划** (`motion_planning.py` + `pick_place_task.py`):IK 求解 + 抓取-搬运-放置状态机
5. **控制 DISCOVERSE**(`client_task_1.py` / `client_task_2.py`):按赛题命名的任务脚本,串联①-④驱动仿真

## 当前阶段的重要说明

**本机尚未安装 DISCOVERSE 仓库**,所以③④⑤都采用"接口对齐真实平台、
但可离线运行"的方式实现:

- ③④ 严格按照 DISCOVERSE 开源示例的真实接口设计(见下方"与 DISCOVERSE 对接"),
  数据结构和函数签名与真实平台一致
- ⑤ 的 `client_task_1.py` / `client_task_2.py` 会自动探测 `discoverse` 是否可
  import:能 import 则走真实仿真分支(`run_real()`);不能则自动切换到离线
  演示分支(`run_dry_run()`),用合成测试图 + 固定占位深度跑通①-④全部逻辑,
  只打印每一步的目标位姿/关节角/夹爪指令,不做真实物理仿真

安装好 DISCOVERSE 并拿到赛题方提供的场景资产后,只需:
1. 替换 `client_task_*.py` 里标 `TODO` 的场景配置(`cfg.mjcf_file_path` 等)
   和 `SimNode.domain_randomization()` / `check_success()`
2. 用赛题方提供的真实相机标定参数替换 `camera_config.py` 里的占位值
3. 用真实场景里工具桶/物料盒的尺寸替换 `grasp_pose.py` 的 `REFERENCE_OBJECT_RADIUS`

业务逻辑(①-④)本身不需要改动。

## 与 DISCOVERSE 对接的关键接口(已核对官方仓库源码)

| 环节 | DISCOVERSE 真实接口 | 本项目对应实现 |
|------|---------------------|----------------|
| 逆运动学 | 官方 `examples/material_sorting/mmk2_kdl.py` 的 NumPy `MMK2Kdl/ArmKdl`，无解返回空解列表 | `motion_planning.MMK2KdlBackend`（单臂/双臂、slide 搜索、限位和失败关闭） |
| 相机观测 | `sim_node.step(action)` 返回 `obs["img"]`(RGB numpy 数组)、`obs["depth"]`(深度 numpy 数组),定义见 `discoverse/robots_env/mmk2_base.py` 的 `getObservation()` | `grounding.locate_object()` 已支持直接传入 numpy 数组,无需先存盘 |
| 控制向量 | `action` 长度 19:0-1 底盘轮速,2 升降,3-4 头部,5-10 左臂,11 左爪,12-17 右臂,18 右爪(见 `mmk2_base.py`) | `pick_place_task.ARM_JOINT_SLICE` / `GRIPPER_INDEX` 常量与之一一对应 |
| 任务基类 | `discoverse.task_base.MMK2TaskBase`,子类实现 `domain_randomization()` / `check_success()`,主循环 `sim_node.reset()` → 循环 `sim_node.step(action)`(参考官方示例 `examples/tasks_mmk2/kiwi_pick.py`) | `client_task_1.py` / `client_task_2.py` 中的 `SimNode` 类(仅在检测到 discoverse 时定义) |

DISCOVERSE 不是 ROS/MoveIt 体系,而是 Python 内嵌 MuJoCo 仿真(控制脚本
`import discoverse` 后在同一进程内直接推进物理步),因此原设计里的
"MoveIt 运动规划"已替换为该平台自带的逆运动学求解器。

## 环境准备

```powershell
pip install -r requirements.txt
```

申请阿里云百炼 API Key(https://bailian.console.aliyun.com/ ,开通后在「API-KEY 管理」创建),然后设置环境变量:

```powershell
# 当前 PowerShell 会话生效
$env:DASHSCOPE_API_KEY="sk-xxxx"

# 或永久生效(需重开终端)
setx DASHSCOPE_API_KEY "sk-xxxx"
```

默认模型为 `qwen3.7-plus`(千问旗舰多模态模型,官方文档确认支持物体定位/grounding),
可用环境变量 `QWEN_VL_MODEL` 切换(如 `qwen3.6-flash` 降低成本)。

与 Qwen2.5-VL 的两个重要差异(代码已按官方文档适配):

- **坐标格式**:qwen3.7 系列返回归一化到 `[0, 999]` 的相对坐标(Qwen2.5-VL 是像素绝对坐标),
  `grounding.py` 中的 `norm_bbox_to_pixel()` 负责按 `值/1000*边长` 换算回像素。
  若换回 Qwen2.5-VL 系列模型,需要去掉这一步换算。
- **思考模式**:qwen3.7 系列默认开启深度思考,本项目通过 `extra_body={"enable_thinking": False}`
  关闭,以降低延迟和 token 消耗(见 `config.py` 的 `EXTRA_BODY`)。

接口默认使用通用域名 `https://dashscope.aliyuncs.com/compatible-mode/v1`;
如需切换到业务空间专属域名,设置环境变量
`DASHSCOPE_BASE_URL="https://llm-ieblg8jrbniinxer.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"`。

## 使用

```powershell
# 生成合成测试图(粉/黄/棕包装盒 + 圆形工具桶)
python make_test_image.py

# 只跑①②:指令解析 + 目标定位 + 参照物定位
python main.py --instruction "找到一个粉色长方体包装盒,放到圆形工具桶左边" --image test_images\boxes.jpg

# 跑①-④全流程(离线演示,未安装 discoverse 时自动降级,不影响运行)
python client_task_1.py
python client_task_2.py
# 任务二可用环境变量临时指定不同指令测试(真实指令来源见 client_task_2.py 文件头 TODO)
$env:TASK2_INSTRUCTION="找到一个黄色长方体包装盒,放到方形物料盒右边"; python client_task_2.py
```

`main.py` 输出:

- 解析出的任务 JSON(目标物体、参照物、方向)
- 目标的边界框 `[x1, y1, x2, y2]` 和中心点 `[cx, cy]`(像素坐标)
- 标注图保存在 `outputs\boxes_annotated.png`,人工核验框的位置

`client_task_*.py` 离线演示模式额外输出:

- ③ 抓取位姿 / 放置位姿的三维坐标(base_link 系)
- ④ 状态机每个阶段(APPROACH_PICK → DESCEND_GRASP → CLOSE_GRIPPER →
  LIFT_AFTER_GRASP → MOVE_TO_PLACE → DESCEND_PLACE → OPEN_GRIPPER → RETREAT)
  对应的手臂关节角(Mock IK,非真实解)和夹爪开合值

也可单独测试某一环节(每个模块的 `__main__` 都带自检断言):

```powershell
python task_parser.py "在场景内找到长方体包装盒放到桌子上"
python grounding.py test_images\boxes.jpg "黄色的包装盒"
python camera_config.py    # ③ 像素+深度 <-> 三维坐标 往返自检
python grasp_pose.py       # ③ 抓取/放置位姿生成自检
python motion_planning.py  # ④ IK 后端接口自检(自动用 Mock)
python pick_place_task.py  # ④ 状态机顺序/指令自检
```

## 任务 JSON 格式

```json
{
  "task_type": "pick_and_place",
  "target_object": {"category": "packing_box", "color": "pink", "shape": "cuboid"},
  "reference_object": {"category": "tool_bucket"},
  "direction": "left",
  "place_target": null,
  "raw_instruction": "找到一个粉色长方体包装盒,放到圆形工具桶左边"
}
```

- `category`: `packing_box` / `material_box` / `tool_bucket` / `table` / `shelf`
- `color`: `pink` / `yellow` / `brown` / null
- `direction`: `left` / `right` / null(相对机器人视角)
- 任务一类指令(无颜色、无参照物)时 `reference_object` 为 null、`place_target` 为 `"table"`

## 文件说明

| 文件 | 作用 |
|------|------|
| `config.py` | 模型名、接口地址、API Key 读取(支持 `.env` 文件) |
| `task_parser.py` | ① 指令 → 结构化 JSON(few-shot 提示词) |
| `grounding.py` | ② 图片(路径/PIL/numpy 均可) + 目标 → 边界框/中心点,画框保存 |
| `camera_config.py` | ③ 相机内外参(占位,需替换真实标定)+ 像素/深度 ↔ 三维坐标转换 |
| `grasp_pose.py` | ③ 三维坐标 → 抓取位姿 / 放置位姿(位置 + 接近姿态) |
| `motion_planning.py` | ④ IK 后端抽象:`DiscoverseIKBackend`(真实)/ `MockIKBackend`(离线) |
| `pick_place_task.py` | ④ 抓取-搬运-放置状态机,产出关节角/夹爪控制向量 |
| `client_common.py` | ①-④共用的驱动逻辑(被 client_task_1/2 复用) |
| `client_task_1.py` | ⑤ 任务一脚本(赛题指定命名),自动探测是否有 discoverse |
| `client_task_2.py` | ⑤ 任务二脚本(赛题指定命名),自动探测是否有 discoverse |
| `main.py` | 只跑①②的演示脚本(适合单独调试视觉部分) |
| `make_test_image.py` | 生成合成测试图 |
| `test_images/` | 测试图片(后续换成仿真相机截图) |
| `outputs/` | 标注了边界框的结果图 |

## 已知占位值一览(拿到真实平台后需替换)

| 位置 | 占位内容 | 替换依据 |
|------|----------|----------|
| `camera_config.py` | 相机视场角、安装位置/俯仰角 | 赛题场景真实相机标定参数 / MJCF 相机定义 |
| `grasp_pose.py` `REFERENCE_OBJECT_RADIUS` | 工具桶/物料盒半径估计值 | 赛题 PDF 未给出,需要真实场景模型尺寸 |
| `client_common.py` `TABLE_PLACE_POSITION_BASE` | 任务一"桌子上"的目标坐标 | 真实场景桌面检测结果或已知固定坐标 |
| `client_task_2.py` `get_instruction()` | 用环境变量兜底获取指令 | 赛题服务端真正的指令下发机制(公开仓库未找到文档) |
| `client_task_*.py` `cfg.mjcf_file_path` / `SimNode` 方法 | 场景资产路径、随机化/成功判定逻辑 | 赛题方提供的场景 mjcf 及评分细则(PDF 第六节) |
