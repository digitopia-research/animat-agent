#!/usr/bin/env bash
# start_server.sh — 一键启动 Animat（自动检查并拉起 MQTT / Redis 依赖）
#
# 用法:
#   ./scripts/start_server.sh          # 启动（若已在跑则跳过）
#   ./scripts/start_server.sh start    # 同上
#   ./scripts/start_server.sh stop     # 停止 animat 与 env_simulator（不动 MQTT/Redis）
#   ./scripts/start_server.sh restart  # stop + start
#   ./scripts/start_server.sh status   # 查看各组件状态
#
# 环境变量:
#   ANIMAT_NO_ENV_SIM=1        只起 animat，不起环境模拟器
#   ANIMAT_NO_WEBUI=1          不起浏览器控制台
#   ANIMAT_WEBUI_PORT=8000     浏览器控制台端口
#   ANIMAT_ENV_SCENE=lively    环境模拟器场景密度 (quiet/normal/lively)
#   ANIMAT_ENV_SEED=42         环境模拟器随机种子（可复现）
#
set -euo pipefail

PROJECT_DIR="/root/Workspace/animat-agent"
LOG_DIR="/tmp"
ANIMAT_LOG="${LOG_DIR}/animat.log"
ENV_SIM_LOG="${LOG_DIR}/env_simulator.log"
WEBUI_LOG="${LOG_DIR}/webui.log"
ANIMAT_BIN="${PROJECT_DIR}/.venv/bin/animat"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
WEBUI_PORT="${ANIMAT_WEBUI_PORT:-8000}"

port_listening() {
    # $1 = 端口号
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${1}\$"
}

animat_running() {
    pgrep -f "animat --reset" >/dev/null 2>&1 || pgrep -f "animat.main:cli" >/dev/null 2>&1
}

env_sim_running() {
    pgrep -f "animat.env_simulator" >/dev/null 2>&1
}

webui_running() {
    pgrep -f "serve_webui.py" >/dev/null 2>&1
}

start_mosquitto() {
    if port_listening 1883; then
        echo "[ok] mosquitto 已在 1883 监听"
        return
    fi
    echo "[..] 启动 mosquitto (MQTT Broker)"
    if command -v mosquitto >/dev/null 2>&1; then
        mosquitto -d
    else
        echo "[!] 未找到 mosquitto，尝试 apt-get 安装"
        apt-get install -y mosquitto >/dev/null 2>&1 || true
        mosquitto -d
    fi
    sleep 1
    if port_listening 1883; then
        echo "[ok] mosquitto 已启动 (1883)"
    else
        echo "[!] mosquitto 启动失败，请手动检查"
    fi
}

start_redis() {
    if port_listening 6379; then
        echo "[ok] redis 已在 6379 监听"
        return
    fi
    echo "[..] 启动 redis-server"
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes
    else
        echo "[!] 未找到 redis-server，请手动安装/启动"
    fi
    sleep 1
    if port_listening 6379; then
        echo "[ok] redis 已启动 (6379)"
    else
        echo "[!] redis 启动失败，请手动检查"
    fi
}

start_animat() {
    if animat_running; then
        echo "[ok] animat 已在运行"
        return
    fi
    cd "${PROJECT_DIR}"
    echo "[..] 启动 animat（本地 GPU 模型）"
    nohup "${ANIMAT_BIN}" --reset > "${ANIMAT_LOG}" 2>&1 &
    # 14B 模型加载约 5s，留足时间再判断，避免误报启动失败
    sleep 6
    if animat_running; then
        echo "[ok] animat 已启动，日志 -> ${ANIMAT_LOG}"
    else
        echo "[!] animat 启动失败，日志末尾："
        tail -n 25 "${ANIMAT_LOG}" 2>/dev/null || true
    fi
}

start_env_sim() {
    if [ "${ANIMAT_NO_ENV_SIM:-0}" = "1" ]; then
        echo "[ok] 已跳过 env_simulator (ANIMAT_NO_ENV_SIM=1)"
        return
    fi
    if env_sim_running; then
        echo "[ok] env_simulator 已在运行"
        return
    fi
    cd "${PROJECT_DIR}"
    echo "[..] 启动 env_simulator（环境世界）"
    # 纯后端进程，不读 stdin、无终端命令。手动注入事件请打开
    #   http://<本机IP>:${WEBUI_PORT}/world.html
    # 该页面通过 MQTT 话题 animat/env/control 下发命令。
    nohup "${PYTHON_BIN}" -m animat.env_simulator \
        > "${ENV_SIM_LOG}" 2>&1 < /dev/null &
    sleep 3
    if env_sim_running; then
        echo "[ok] env_simulator 已启动，日志 -> ${ENV_SIM_LOG}"
    else
        echo "[!] env_simulator 启动失败，日志末尾："
        tail -n 20 "${ENV_SIM_LOG}" 2>/dev/null || true
    fi
}

stop_env_sim() {
    if ! env_sim_running; then
        echo "[ok] env_simulator 未运行"
        return
    fi
    echo "[..] 停止 env_simulator"
    pkill -f "animat.env_simulator" 2>/dev/null || true

    local waited=0
    while env_sim_running && [ "${waited}" -lt 10 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if env_sim_running; then
        pkill -9 -f "animat.env_simulator" 2>/dev/null || true
        sleep 1
    fi
    env_sim_running && echo "[!] env_simulator 仍未停止" || echo "[ok] env_simulator 已停止"
}

start_webui() {
    if [ "${ANIMAT_NO_WEBUI:-0}" = "1" ]; then
        echo "[ok] 已跳过 webui (ANIMAT_NO_WEBUI=1)"
        return
    fi
    if webui_running; then
        echo "[ok] webui 已在运行 (端口 ${WEBUI_PORT})"
        return
    fi
    cd "${PROJECT_DIR}"
    echo "[..] 启动 webui（浏览器控制台）"
    ANIMAT_WEBUI_PORT="${WEBUI_PORT}" nohup "${PYTHON_BIN}" scripts/serve_webui.py \
        > "${WEBUI_LOG}" 2>&1 < /dev/null &
    sleep 2
    if webui_running; then
        echo "[ok] webui 已启动 -> http://<本机IP>:${WEBUI_PORT}/  日志 ${WEBUI_LOG}"
    else
        echo "[!] webui 启动失败，日志末尾："
        tail -n 15 "${WEBUI_LOG}" 2>/dev/null || true
    fi
}

stop_webui() {
    if ! webui_running; then
        echo "[ok] webui 未运行"
        return
    fi
    echo "[..] 停止 webui"
    pkill -f "serve_webui.py" 2>/dev/null || true
    sleep 1
    webui_running && pkill -9 -f "serve_webui.py" 2>/dev/null || true
    webui_running && echo "[!] webui 仍未停止" || echo "[ok] webui 已停止"
}

stop_animat() {
    if ! animat_running; then
        echo "[ok] animat 未运行"
        return
    fi

    echo "[..] 停止 animat"
    # 先礼后兵：SIGTERM 让进程走优雅关闭（保存状态、断开 MQTT/Redis、
    # 释放 GPU 显存）。14B 模型释放显存需要数秒，固定 sleep 1 不够，
    # 会导致 restart 里 start 判定"仍在运行"而跳过启动——表现为永远重启不了。
    pkill -f "animat --reset" 2>/dev/null || true
    pkill -f "animat.main:cli" 2>/dev/null || true

    local waited=0
    local timeout="${ANIMAT_STOP_TIMEOUT:-20}"
    while animat_running && [ "${waited}" -lt "${timeout}" ]; do
        sleep 1
        waited=$((waited + 1))
    done

    if animat_running; then
        echo "[!] ${timeout}s 内未退出，强制终止"
        pkill -9 -f "animat --reset" 2>/dev/null || true
        pkill -9 -f "animat.main:cli" 2>/dev/null || true
        sleep 2
    fi

    # 强制终止后仍需等待显存释放，否则新进程加载模型会 OOM
    if animat_running; then
        echo "[!] 仍未停止"
    else
        echo "[ok] animat 已停止"
        if [ -n "${ANIMAT_STOP_GRACE:-}" ]; then
            sleep "${ANIMAT_STOP_GRACE}"
        else
            sleep 3
        fi
    fi
}

show_status() {
    echo "=== 组件状态 ==="
    port_listening 1883 && echo "mosquitto (1883): 监听中" || echo "mosquitto (1883): 未监听"
    port_listening 6379 && echo "redis (6379):     监听中" || echo "redis (6379):     未监听"
    animat_running && echo "animat:          运行中" || echo "animat:          未运行"
    env_sim_running && echo "env_simulator:   运行中" || echo "env_simulator:   未运行"
    webui_running && echo "webui:           运行中 (端口 ${WEBUI_PORT})" || echo "webui:           未运行"
}

case "${1:-start}" in
    start)   start_mosquitto; start_redis; start_animat; start_env_sim; start_webui; echo "--- 完成 ---" ;;
    stop)    stop_animat; stop_env_sim; stop_webui ;;
    restart) stop_animat; stop_env_sim; stop_webui; start_mosquitto; start_redis; start_animat; start_env_sim; start_webui; echo "--- 完成 ---" ;;
    status)  show_status ;;
    *) echo "用法: $0 {start|stop|restart|status}"; exit 1 ;;
esac
