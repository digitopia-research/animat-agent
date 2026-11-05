# Bot 模拟器使用手册（身体）

> 适用模块：`animat/simulators/bot_simulator.py`
> 网页控制台：`webui/body.html`
> 适用版本：Animat Agent v2（MQTT + Redis 双进程架构）

---

## 1. 这是什么

`bot_simulator` 模拟**这台机器人自身的生理状态**：电量、体温、充电。

它不模拟外部世界。昼夜、光照、路人、触碰、噪声全部由
`animat/simulators/env_simulator.py` 负责，详见
`docs/environment_simulator_cn.md`。

两者都往 `animat/sensor/interoception` 写数据，但写的字段互不重叠：

| | `bot_simulator` | `env_simulator` |
|---|---|---|
| 角色 | 身体 | 世界 |
| 发布字段 | `battery_pct` / `temperature` / `charging` / `imu_tilt_deg` | `ambient_light` / `ambient_db` / `event_text` / `load` |
| 网页 | `body.html` | `world.html` |
| 启动开关 | `ANIMAT_NO_BOT_SIM=1` | `ANIMAT_NO_ENV_SIM=1` |

> **为什么要拆开**：早先两个进程各自算电量和体温，每 2 秒交替覆盖，
> Animat 收到的数值在两种模型之间来回跳。你在网页上把电量改到 15%，
> 下一帧可能被另一个进程刷回 99%，且从外部完全看不出是谁写的。
> 现在身体量只有一个权威来源，就是本模块。

---

## 2. 启动

正常部署由脚本统一管理，不需要单独启动：

```bash
./scripts/start_server.sh start      # 含 bot_simulator
./scripts/start_server.sh status     # 查看五个组件
./scripts/start_server.sh stop
```

单独运行（调试时）：

```bash
uv run python -m animat.simulators.bot_simulator
```

> ⚠️ 不要用系统 python 或 conda 的 python，它们通常没装项目依赖。

日志位置：`/tmp/bot_simulator.log`

---

## 3. 控制方式

**没有终端交互。** 这是后端服务，不读 stdin。

所有操作走 MQTT 或浏览器：

```
http://<本机IP>:8000/body.html
```

页面提供：电量环形仪表、充电/停止/重置按钮、电量/体温/放电速率三个滑块。

### 3.1 MQTT 命令

发布到 `animat/body/control`：

```bash
# 开始充电
mosquitto_pub -h 127.0.0.1 -t animat/body/control -m '{"cmd":"charge"}'

# 设定电量 15%
mosquitto_pub -h 127.0.0.1 -t animat/body/control -m '{"cmd":"scale","arg":"15"}'

# 设定体温 39.5°C
mosquitto_pub -h 127.0.0.1 -t animat/body/control -m '{"cmd":"temp","arg":"39.5"}'
```

| cmd | arg | 说明 |
|---|---|---|
| `charge` | — | 开始充电 |
| `stop` | — | 停止充电 |
| `reset` | — | 电量重置到理想值 |
| `status` | — | 查询当前状态 |
| `scale` | 0-100 | 设定电量百分比 |
| `temp` | 20-60 | 设定体温 °C |
| `discharge` | 0-10 | 设定放电速率 %/周期 |
| `help` | — | 列出可用命令 |

执行结果发布在 `animat/body/status`，形如：

```json
{"type": "ack", "cmd": "scale", "ok": true, "msg": "电量已设为 15%"}
```

状态快照（每 2 秒广播一次）：

```json
{"type": "status", "battery_pct": 14.7, "temperature": 37.0,
 "charging": false, "discharge_rate": 0.05, "imu_tilt_deg": -0.5}
```

---

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ANIMAT_SIM_BATTERY` | `100` | 初始电量 % |
| `ANIMAT_SIM_DISCHARGE` | `0.05` | 放电速率 %/周期（2 秒一次） |
| `MQTT_BROKER_HOST` | `localhost` | broker 地址 |
| `MQTT_BROKER_PORT` | `1883` | broker 端口 |

示例：让电量从满电快速掉到危险线，观察 Animat 的生存焦虑：

```bash
ANIMAT_SIM_BATTERY=100 ANIMAT_SIM_DISCHARGE=0.5 \
  uv run python -m animat.simulators.bot_simulator
```

---

## 5. 物理因果

以下几处不是随机的，而是刻意建模的因果关系：

**充电升温**：充电时体温上升到 41°C 封顶，停止后缓慢回落到 37°C。
封顶是必要的——早前放到 45°C 时，模型会编造"系统过热"的叙事。

**走到充电位即充电**：订阅 `animat/motor`，收到
`{"command":"move","params":{"direction":"charger"}}` 时自动开始充电；
离开充电位则断电。这条逻辑放在身体模块，因为它修改的是电池状态。

**满电自动停充**：充到 100% 自动关闭充电开关。

---

## 6. 常见问题

**Q：页面显示「模拟器离线」？**

`bot_simulator` 没在跑。检查：

```bash
./scripts/start_server.sh status
tail -n 30 /tmp/bot_simulator.log
```

**Q：改了电量但 Animat 没反应？**

内感每 2 秒上报一次，等一轮即可。如果仍然不更新，确认 `animat` 主进程
在运行，且没有另一个进程在覆盖同一话题：

```bash
mosquitto_sub -h 127.0.0.1 -t animat/sensor/interoception -v
```

正常情况下应该在两种报文之间交替——一种只有环境量（`ambient_light`），
一种只有生理量（`battery_pct`）。如果两种报文都带 `battery_pct`，
说明跑的是旧版本代码。

**Q：`uv run python -m animat.simulators.bot_simulator` 报 ModuleNotFoundError？**

用了系统 python。改用 `uv run`，或确认 `.venv` 已激活。
