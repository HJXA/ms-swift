#!/bin/bash

while true; do
    echo "$(date '+%F %T')"
    echo "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    sleep 1
done