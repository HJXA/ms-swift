#!/bin/bash

# ==========================================
# 内存监控与二级精准终止脚本
# ==========================================

THRESHOLD=95
START_PID=3981025
END_PID=3981028
TARGET_USER="jxhe"
PATTERN="swift/cli/pt.py"
TMUX_SESSION="1"

# 标记变量：记录 Python 进程是否已经被清理过
PYTHON_CLEANED=false

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 高级监控启动... 内存阈值: ${THRESHOLD}%"

while true; do
    NOW=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 获取内存使用百分比
    RAM_USAGE=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')
    echo "[$NOW] 当前内存使用率: ${RAM_USAGE}%"

    # 使用 bc 进行浮点数比较
    if [ "$(echo "$RAM_USAGE >= $THRESHOLD" | bc -l)" -eq 1 ]; then
        echo "[$NOW] [!] 警告: 内存已达临界值 ${RAM_USAGE}%!"

        if [ "$PYTHON_CLEANED" = false ]; then
            # --- 第一阶段：清理 Python 进程 ---
            echo "[$NOW] 执行第一阶段：清理 Python 进程..."
            for pid in $(seq $START_PID $END_PID); do
                PROCESS_INFO=$(ps -p $pid -u "$TARGET_USER" -o args= 2>/dev/null)
                if [[ "$PROCESS_INFO" == *"$PATTERN"* ]]; then
                    echo "[$NOW] [*] 正在终止进程: $pid"
                    kill -9 $pid
                fi
            done
            PYTHON_CLEANED=true
            echo "[$NOW] Python 进程清理尝试完成。"
        else
            # --- 第二阶段：如果清理过 Python 内存仍超标，干掉 Tmux 会话 ---
            echo "[$NOW] 警告：清理 Python 后内存依然超标！执行第二阶段：强制关闭 Tmux 会话 $TMUX_SESSION"
            
            # 检查 tmux 会话是否存在
            if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
                tmux kill-session -t "$TMUX_SESSION"
                echo "[$NOW] [√] 已强制关闭 Tmux 会话 $TMUX_SESSION。"
            else
                echo "[$NOW] [!] 未发现运行中的 Tmux 会话 $TMUX_SESSION。"
            fi
            
            echo "[$NOW] 所有清理策略已执行，监控任务结束。"
            exit 0
        fi
    fi

    # 内存高时建议缩短检查间隔，比如改为 10 秒
    sleep 60
done