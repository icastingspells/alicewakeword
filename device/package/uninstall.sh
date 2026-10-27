#!/bin/sh
set -eu
[ "$(id -u)" = 0 ]
if [ -x /data/eva/current/evactl ];then /data/eva/current/evactl disable;fi
remounted=0
trap 'if [ "$remounted" = 1 ];then mount -o remount,ro /;fi' EXIT
if [ -e /etc/init.d/S98eva ];then
    grep -q '^# EVA_MANAGED_INIT_V1$' /etc/init.d/S98eva || { echo 'Unknown init entry: refusing removal';exit 1; }
    if awk '$2=="/" && $4 ~ /(^|,)ro(,|$)/ {found=1} END {exit !found}' /proc/mounts;then mount -o remount,rw /;remounted=1;fi
    rm -f /etc/init.d/S98eva
fi
echo 'EVA autostart removed. Original launch.json restored; /data/eva releases retained for rollback.'
