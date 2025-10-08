#!/bin/bash

#################### 配置区 ####################
# 基础参数（修改这些变量）
tracker_name=spikefet
dataset_name=visevent
tracker_param=spikefet_visevent_tiny

# 实验控制
MAX_RUNS=0        # 最大运行次数（0表示无限）
INTERVAL=0     # 实验间隔（秒）
LOG_DIR="./logs"  # 日志保存目录

# 检查点监控配置
CHECK_DIR="output/checkpoints/train/spikefet/${tracker_param}"  # 监控的目录
CHECK_FILE="SpikeFET_ep0050.pth.tar"  # 需要等待的文件名（根据实际名称修改）
SLEEP_INTERVAL=30  # 文件检查间隔（秒）

#################### 执行逻辑 ####################
# 创建日志目录
mkdir -p "$LOG_DIR"

# 等待关键文件生成
echo "等待检查点文件生成: $CHECK_DIR/$CHECK_FILE"
while [ ! -f "$CHECK_DIR/$CHECK_FILE" ]; do
    sleep $SLEEP_INTERVAL
done
echo "检测到文件存在，开始执行实验！"

run_counter=0

# 主循环
for EPOCH in {40..50}; do
#for EPOCH in 40 45 50; do
    # 打印当前实验信息
    echo "当前实验：EPOCH=$EPOCH"
    # 生成唯一实验ID
    EXP_ID=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/exp_${EXP_ID}.log"
    padded_num=$(printf "%04d" $EPOCH)
    ckpt_path="output/checkpoints/train/spikefet/${tracker_param}/SpikeFET_ep${padded_num}.pth.tar"

    # 构造命令
    cmd="python tracking/test.py $tracker_name $tracker_param --dataset $dataset_name --threads 4 --num_gpus 2 --ckpt $ckpt_path"

    # 执行命令
    echo "启动实验 $EXP_ID"
    echo "完整命令: $cmd"

    # 使用tee同时输出到文件和终端
    eval $cmd 2>&1 | tee "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "实验 $EXP_ID 失败！错误日志：$LOG_FILE"
        # 是否继续执行取决于需求
        # exit 1  # 遇到错误立即退出
    fi

    # 运行次数控制
    ((run_counter++))
    if [ $MAX_RUNS -ne 0 ] && [ $run_counter -ge $MAX_RUNS ]; then
        echo "达到最大运行次数 $MAX_RUNS"
        exit 0
    fi

    # 间隔等待
    echo "等待 $INTERVAL 秒后继续..."
    sleep $INTERVAL
done