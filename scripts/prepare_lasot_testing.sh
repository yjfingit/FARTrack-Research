#!/usr/bin/env bash
set -euo pipefail

# Official LaSOT testing archive only. The script never writes inside the
# archive and extracts into the data disk configured by lib/test/evaluation/local.py.
archive="${1:-/root/autodl-tmp/experiment/.research-assets/downloads/LaSOTTesting.zip}"
data_root="${2:-/root/autodl-tmp/experiment/.research-assets/data/lasot}"
expected_md5="a9038384fd94d30e7ad6a1b7cf32ec73"

test -f "$archive"
actual_md5="$(md5sum "$archive" | awk '{print $1}')"
test "$actual_md5" = "$expected_md5"

mkdir -p "$data_root"
unzip -q -n "$archive" -d "$data_root"
find "$data_root" -type f -name testing_set.txt -print -quit | grep -q . || {
  echo "testing_set.txt not found after extraction" >&2
  exit 1
}
