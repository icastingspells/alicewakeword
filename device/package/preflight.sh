#!/bin/sh
# Read-only, credential-free compatibility and conflict checks.
set -eu
BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$BASE/profile.env"
failed=0
check() { name=$1; shift; if "$@"; then echo "$name=ok"; else echo "$name=FAIL"; failed=1; fi; }
echo EVA_PREFLIGHT_VERSION=1
check ROOT test "$(id -u)" = 0
check DATA_DIRECTORY test -d /data
if [ -L /data/eva ]; then echo INSTALL_ROOT_SYMLINK=FAIL;failed=1;fi
actual=$(sha256sum /system/vendor/quasar/maind 2>/dev/null | awk '{print $1}')
check FIRMWARE_HASH test "$actual" = "$FIRMWARE_SHA256"
elf=$(od -An -tu1 -j4 -N1 /system/vendor/quasar/maind 2>/dev/null | tr -d ' ')
check USERSPACE_ELF32 test "$elf" = 1
launch=$(sha256sum /system/vendor/quasar/launch.json 2>/dev/null | awk '{print $1}')
if [ "$launch" = "$BASE_LAUNCH_SHA256" ] || [ "$launch" = "$PATCHED_LAUNCH_SHA256" ]; then echo LAUNCH_HASH=ok;else echo LAUNCH_HASH=FAIL;failed=1;fi
available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
echo MEM_AVAILABLE_KB=${available:-unknown}
check MEMORY_HEADROOM test "${available:-0}" -ge "$MIN_AVAILABLE_KB"
free=$(df -Pk /data 2>/dev/null | awk 'END {print $4}')
echo DATA_FREE_KB=${free:-unknown}
check FLASH_HEADROOM test "${free:-0}" -ge "$MIN_DISK_KB"
owner=$(awk '/owner_pid/ {print $3}' /proc/asound/card0/pcm1c/sub0/status 2>/dev/null || true)
case "$owner" in ''|*[!0-9]*) echo CAPTURE_OWNER=not_available ;;
*) executable=$(readlink /proc/$owner/exe 2>/dev/null || true)
   if [ "$executable" = /system/vendor/quasar/maind ];then echo CAPTURE_OWNER=quasar;else echo CAPTURE_OWNER=OTHER_PROCESS;failed=1;fi ;;
esac
if [ -e /etc/init.d/S98eva ] && ! grep -q '^# EVA_MANAGED_INIT_V1$' /etc/init.d/S98eva;then echo INIT_CONFLICT=FAIL;failed=1;fi
for pid in $(pidof maind 2>/dev/null || true);do
    if tr '\000' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q -- '--platformd';then
        preload=$(tr '\000' '\n' </proc/$pid/environ 2>/dev/null | sed -n 's/^LD_PRELOAD=//p')
        case "$preload" in ''|/data/eva/current/libeva_tap.so) echo PROCESS_PRELOAD=ok;;*) echo PROCESS_PRELOAD=CONFLICT;failed=1;;esac
    fi
done
echo ACTIVATION_DEFAULT=observe
echo MUTE_AND_PCM_ON_PHYSICAL_DEVICE=not_verified
echo CLI_ON_LITE2_RUNTIME=not_verified
exit "$failed"
