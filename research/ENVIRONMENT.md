# Reproducible Research Environment

The research runs use the following isolated environment. All package downloads
and caches are on the data disk, not the system disk.

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research
PIP_CACHE_DIR=/root/autodl-tmp/experiment/.pip-cache \
  "$VENV/bin/python" -m pip install safetensors tzdata
"$VENV/bin/python" -m pip check
```

The verification after installation returned `No broken requirements found`.
The training launcher exports the following limits before importing numerical
libraries or creating its four DataLoader workers:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
FARTRACK_TORCH_THREADS=1
FARTRACK_TORCH_INTEROP_THREADS=1
FARTRACK_OPENCV_THREADS=0
```

These caps prevent OpenMP/BLAS/OpenCV thread multiplication across worker
processes. They are a throughput-safety setting, not a change to model
training, data, tracker, or evaluation protocol.
