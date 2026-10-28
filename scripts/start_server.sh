#!/usr/bin/env bash
# start_server.sh — 一键启动 Animat（自动检查并拉起 MQTT / Redis 依赖）
#
# 用法:
#   ./scripts/start_server.sh          # 启动（若已在跑则跳过）
#   ./scripts/start_server.sh start    # 同上
#   ./scripts/start_server.sh stop     # 停止 animat（不动 MQTT/Redis）
#   ./scripts/start_server.sh restart  # stop + start
#   ./scripts/start_server.sh status   # 查看各组件状态
#
set -euo pipefail

PROJECT_DIR="/root/Workspace/animat-agent"
LOG_DIR="/tmp"
ANIMAT_LOG="${LOG_DIR}/animat.log"
ANIMAT_BIN="${PROJECT_DIR}/.venv/bin/animat"

port_listening() {
    # $1 = 端口号
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${1}\$"
}

animat_running() {
    pgrep -f "animat --reset" >/dev/null 2>&1 || pgrep -f "animat.main:cli" >/dev/null 2>&1
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
    sleep 3
    if animat_running; then
        echo "[ok] animat 已启动，日志 -> ${ANIMAT_LOG}"
    else
        echo "[!] animat 启动失败，日志末尾："
        tail -n 25 "${ANIMAT_LOG}" 2>/dev/null || true
    fi
}

stop_animat() {
    if animat_running; then
        echo "[..] 停止 animat"
        pkill -f "animat --reset" 2>/dev/null || true
        pkill -f "animat.main:cli" 2>/dev/null || true
        sleep 1
        animat_running && echo "[!] 仍未停止" || echo "[ok] animat 已停止"
    else
        echo "[ok] animat 未运行"
    fi
}

show_status() {
    echo "=== 组件状态 ==="
    port_listening 1883 && echo "mosquitto (1883): 监听中" || echo "mosquitto (1883): 未监听"
    port_listening 6379 && echo "redis (6379):     监听中" || echo "redis (6379):     未监听"
    animat_running && echo "animat:          运行中" || echo "animat:          未运行"
}

case "${1:-start}" in
    start)   start_mosquitto; start_redis; start_animat; echo "--- 完成 ---" ;;
    stop)    stop_animat ;;
    restart) stop_animat; start_mosquitto; start_redis; start_animat; echo "--- 完成 ---" ;;
    status)  show_status ;;
    *) echo "用法: $0 {start|stop|restart|status}"; exit 1 ;;
esac
