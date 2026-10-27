#!/bin/sh
set -eu
# The small detector's address-space limit must not prevent the stock CLI loading.
ulimit -S -v unlimited
# Fixed native CLI command; no model output or user text is evaluated as shell code.
BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$BASE/profile.env"
[ "$(sha256sum /system/vendor/quasar/maind | awk '{print $1}')" = "$FIRMWARE_SHA256" ]
# Quasar arbitrates the request itself. Its logs may rotate or omit state records.
mute=$("$BASE/eva-native" --mic-state) || { echo 'EVA_ACTIVATION_SKIPPED reason=mic_state_unavailable';exit 0; }
[ "$mute" = MICS_MUTED=0 ] || { echo 'EVA_ACTIVATION_SKIPPED reason=muted';exit 0; }
[ ! -L /tmp/eva ] || exit 1
mkdir /tmp/eva/activation.lock 2>/dev/null || { echo 'EVA_ACTIVATION_SKIPPED reason=locked';exit 0; }
trap 'rmdir /tmp/eva/activation.lock 2>/dev/null || true' EXIT
"$BASE/eva-native" --activate-cli
