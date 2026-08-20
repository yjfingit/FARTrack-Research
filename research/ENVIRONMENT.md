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
An archived, never-invoked CTQA training launcher exported the following limits
before importing numerical libraries or creating its four DataLoader workers:

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
processes. They are retained as environment provenance only; the final research
route is training-free and will not invoke that launcher.

## Archived, Unused Training-Mirror Provenance

The unused GOT-10k training mirror is stored under
`/root/autodl-tmp/experiment/.research-assets/datasets/got10k-train-mirror`.
It is a public Hugging Face mirror (`jinyoungkim/GoT-10k`) containing only the
two uploaded archive parts and no dataset-card metadata; it is not represented
as an official GOT-10k distribution. It was subject to the recorded part
hashes, archive integrity check, and a subsequent training-root inspection.
Before extraction, validate both supplied split-file digests and the joined ZIP
directory with:

```bash
bash scripts/verify_got10k_train_mirror.sh
```

The script intentionally does not remove downloaded parts or extract files.
The archive layout and training-only root were inspected after this gate. The
final method does not use this dataset or the registered CTQA launcher.

After the joined archive passes `unzip -tq`, the two verified, re-downloadable
parts may be removed to recover approximately 70.7 GB before extraction. This
is the only planned deletion in data preparation; it must be logged with the
verified digests and archive path. The joined archive is retained until the
training-only tree has passed structural inspection.
