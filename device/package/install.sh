#!/bin/sh
set -eu
umask 077
SOURCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$SOURCE"
sha256sum -c SHA256SUMS
. ./profile.env
./preflight.sh
./eva-native --self-test
ROOT=/data/eva
[ ! -L "$ROOT" ] || exit 1
[ ! -L "$ROOT/releases" ] || exit 1
mkdir -p "$ROOT/releases";chmod 700 "$ROOT" "$ROOT/releases"
DEST=$ROOT/releases/$RELEASE_ID
if [ -e "$DEST" ];then
    [ "$(sha256sum "$DEST/SHA256SUMS" | awk '{print $1}')" = "$(sha256sum "$SOURCE/SHA256SUMS" | awk '{print $1}')" ] || { echo 'Existing release differs from incoming package';exit 1; }
    (cd "$DEST";sha256sum -c SHA256SUMS)
else
    STAGE=$ROOT/releases/.stage-$RELEASE_ID-$$
    mkdir "$STAGE"
    for file in eva-native libeva_tap.so evactl preflight.sh activate.sh profile.env launch.json S98eva SHA256SUMS config.env install.sh uninstall.sh models.json LICENSE LICENSE.openwakeword NOTICE;do
        cp "$SOURCE/$file" "$STAGE/$file"
    done
    chmod 700 "$STAGE/eva-native" "$STAGE/evactl" "$STAGE/preflight.sh" "$STAGE/activate.sh" "$STAGE/S98eva"
    (cd "$STAGE";sha256sum -c SHA256SUMS)
    mv "$STAGE" "$DEST"
fi
if [ -e "$ROOT/current" ] && [ ! -L "$ROOT/current" ];then echo 'Unexpected current path';exit 1;fi
if [ -L "$ROOT/current" ];then "$ROOT/current/evactl" stop;readlink "$ROOT/current" >"$ROOT/previous";fi
# Every install/update returns to observation and invalidates the old timing result.
cp "$DEST/config.env" "$ROOT/config.env";chmod 600 "$ROOT/config.env"
rm -f "$ROOT/measurement.env"
if [ -L "$ROOT/current.new" ];then rm -f "$ROOT/current.new";fi
[ ! -e "$ROOT/current.new" ] || exit 1
ln -s "releases/$RELEASE_ID" "$ROOT/current.new"
# Firmware BusyBox mv lacks -T; POSIX rename replaces the symlink atomically.
"$DEST/eva-native" --publish-link
remounted=0
restore_mount() { if [ "$remounted" = 1 ];then mount -o remount,ro /;fi; }
trap restore_mount EXIT
if awk '$2=="/" && $4 ~ /(^|,)ro(,|$)/ {found=1} END {exit !found}' /proc/mounts;then mount -o remount,rw /;remounted=1;fi
cp "$DEST/S98eva" /etc/init.d/S98eva.new;chmod 755 /etc/init.d/S98eva.new;mv -f /etc/init.d/S98eva.new /etc/init.d/S98eva
echo "EVA_INSTALLED=$RELEASE_ID"
echo 'Autostart installed in observe mode. Reboot to attach tap; Quasar was not restarted.'
