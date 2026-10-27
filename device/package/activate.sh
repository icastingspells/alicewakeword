#!/bin/sh
set -eu
# The small detector's address-space limit must not prevent the stock CLI loading.
ulimit -S -v unlimited
# Fixed native CLI command; no model output or user text is evaluated as shell code.
BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$BASE/profile.env"
[ "$(sha256sum /system/vendor/quasar/maind | awk '{print $1}')" = "$FIRMWARE_SHA256" ]
log=/data/quasar/daemons_logs/maind.log
[ -r "$log" ] || exit 0
state=$(grep 'onAliceStateChanged:' "$log" | tail -n 1 | sed 's/.*onAliceStateChanged: //')
[ "$state" = IDLE ] || exit 0
mute=$("$BASE/eva-native" --mic-state) || exit 0
[ "$mute" = MICS_MUTED=0 ] || exit 0
[ ! -L /tmp/eva ] || exit 1
mkdir /tmp/eva/activation.lock 2>/dev/null || exit 0
trap 'rmdir /tmp/eva/activation.lock 2>/dev/null || true' EXIT
"$BASE/eva-native" --activate-cli
