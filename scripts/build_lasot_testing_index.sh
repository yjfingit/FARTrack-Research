#!/usr/bin/env bash
set -euo pipefail

# Make a symlink-only category/sequence view for the legacy LaSOT loader.
# The official archive extraction is never modified or copied.
source_root="${1:-/root/autodl-tmp/experiment/.research-assets/data/lasot/LaSOTTesting}"
index_root="${2:-/root/autodl-tmp/experiment/.research-assets/index/lasot_testing}"

test -f "$source_root/testing_set.txt"
mkdir -p "$index_root"

while IFS= read -r sequence || [ -n "$sequence" ]; do
    sequence="${sequence%$'\r'}"
    test -n "$sequence"
    category="${sequence%%-*}"
    source="$source_root/$sequence"
    target_dir="$index_root/$category"
    target="$target_dir/$sequence"
    test -d "$source"
    mkdir -p "$target_dir"
    if [ -L "$target" ]; then
        test "$(readlink -f "$target")" = "$source"
    elif [ -e "$target" ]; then
        echo "refusing to replace non-symlink index entry: $target" >&2
        exit 1
    else
        ln -s "$source" "$target"
    fi
done < "$source_root/testing_set.txt"

test "$(find "$index_root" -type l | wc -l)" -eq 280
