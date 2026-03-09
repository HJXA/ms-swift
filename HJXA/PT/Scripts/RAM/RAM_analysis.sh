# Data Shuffle会导致内存增长，监控内存增长情况

old_mem=$(ps -eo pid,rss --no-headers)
while true; do
  
  new_mem=$(ps -eo pid,rss --no-headers)
  echo "--- $(date +%H:%M:%S) 内存增长检测 ---"
  echo -e "PID\t增长量(KB)\t进程命令"
  
  while read -r pid rss; do
    old_rss=$(echo "$old_mem" | awk -v p=$pid '$1==p {print $2}')
    if [ -n "$old_rss" ] && [ "$rss" -gt "$old_rss" ]; then
      diff=$((rss - old_rss))
      cmd=$(ps -p $pid -o comm=)
      echo -e "$pid\t+$diff\t\t$cmd"
    fi
  done <<< "$new_mem"
  
  old_mem="$new_mem"
  sleep 5
done