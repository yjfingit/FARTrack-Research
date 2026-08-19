#!/usr/bin/env bash
# Verify the public mirror parts before any training-only extraction. This does
# not delete parts or extract data; both operations require a separate review.
set -euo pipefail

root="${1:-/root/autodl-tmp/experiment/.research-assets/datasets/got10k-train-mirror}"
part_aa="${root}/full_data.zip.partaa"
part_ab="${root}/full_data.zip.partab"
archive="${root}/full_data.zip"
temporary_archive="${archive}.partial"

expected_aa="037bda7d6c4ddee0df781d04f6f0688ee861d7f5ab66547610689dfffbfb0988"
expected_ab="72dd314c880f73a3e167354c205d6184fb9e99b1d0041e24b8dad92f26b9fc4d"

[[ -f "${part_aa}" && -f "${part_ab}" ]] || {
  echo "Missing mirror parts under ${root}" >&2
  exit 2
}

[[ "$(sha256sum "${part_aa}" | awk '{print $1}')" == "${expected_aa}" ]] || {
  echo "partaa SHA-256 mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${part_ab}" | awk '{print $1}')" == "${expected_ab}" ]] || {
  echo "partab SHA-256 mismatch" >&2
  exit 1
}

if [[ ! -f "${archive}" ]]; then
  [[ ! -e "${temporary_archive}" ]] || {
    echo "Refusing to overwrite existing partial archive: ${temporary_archive}" >&2
    exit 1
  }
  cat "${part_aa}" "${part_ab}" > "${temporary_archive}"
  mv "${temporary_archive}" "${archive}"
fi

unzip -tq "${archive}"
echo "Verified archive: ${archive}"
echo "Top-level archive entries:"
unzip -Z1 "${archive}" | awk -F/ 'NF {print $1}' | sort -u
