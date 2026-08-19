#!/usr/bin/env bash
# Pre-registered single-GPU CTQA/control screen. Do not launch until the
# coordinator has verified GOT-10k training extraction and checksum evidence.
set -euo pipefail

if [[ "${CTQA_DATA_VERIFIED:-0}" != "1" ]]; then
  echo "Refusing to train: set CTQA_DATA_VERIFIED=1 only after coordinator verification." >&2
  exit 2
fi

if [[ $# -ne 2 || ( "$1" != "control" && "$1" != "ctqa" ) ]]; then
  echo "Usage: $0 {control|ctqa} {seed}" >&2
  exit 2
fi

arm="$1"
seed="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train_root="/root/autodl-tmp/experiment/.research-assets/data/got10k/train"
save_root="/root/autodl-tmp/experiment/.research-assets/runs/ctqa_screen/${arm}/seed_${seed}"
config="fartrack_sparse_224_${arm}_got10k_screen"
venv_python="/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python"
local_file="${root}/lib/train/admin/local.py"
environment_file="${root}/research/ctqa_got10k_train_environment.py"
backup="${local_file}.ctqa-screen-$$"

[[ -d "${train_root}" ]] || { echo "Missing training root: ${train_root}" >&2; exit 2; }
[[ -x "${venv_python}" ]] || { echo "Missing venv: ${venv_python}" >&2; exit 2; }
mkdir -p "${save_root}"

cp "${local_file}" "${backup}"
cleanup() { mv -f "${backup}" "${local_file}"; }
trap cleanup EXIT
cp "${environment_file}" "${local_file}"

cd "${root}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
FARTRACK_TORCH_THREADS=1 FARTRACK_TORCH_INTEROP_THREADS=1 FARTRACK_OPENCV_THREADS=0 \
CUDA_VISIBLE_DEVICES=0 "${venv_python}" lib/train/run_training.py \
  --script fartrack_sparse --config "${config}" --save_dir "${save_root}" \
  --seed "${seed}" --use_lmdb 0 2>&1 | tee "${save_root}/stdout.log"
