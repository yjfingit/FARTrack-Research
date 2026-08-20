#!/usr/bin/env python3
"""Read-only provenance audit for cached cross-model tracker sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


MODELS = {
    "siamrpnpp": {
        "cache": "PySOT",
        "official_remote": "https://github.com/STVIR/pysot.git",
        "family": "classic Siamese correlation",
        "checkpoint_source": "PySOT MODEL_ZOO.md official Google Drive link",
        "inference_hint": "tools/test.py with siamrpn_r50_l234_dwxcorr config",
    },
    "ostrack": {
        "cache": "OSTrack",
        "official_remote": "https://github.com/botaoye/OSTrack.git",
        "family": "one-stream Transformer",
        "checkpoint_source": "official OSTrack README Google Drive model folder",
        "inference_hint": "tracking/test.py ostrack <config> --dataset got10k_val",
    },
    "mixformer": {
        "cache": "MixFormerV2",
        "official_remote": "https://github.com/mcg-nju/MixFormerV2.git",
        "family": "mixed-attention online templates",
        "checkpoint_source": "official MixFormerV2 README Google Drive or NJU Box",
        "inference_hint": "official tracking/test_mixformer.sh-compatible entry point",
    },
    "odtrack": {
        "cache": "ODTrack",
        "official_remote": "https://github.com/GXNU-ZhongLab/ODTrack.git",
        "family": "online dense temporal tokens",
        "checkpoint_source": "official ODTrack README Google Drive model folder",
        "inference_hint": "tracking/test.py odtrack <config> --dataset got10k_val",
    },
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def normalize_remote(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").lower()


def checkpoint_records(root: Path) -> list[dict[str, object]]:
    files = sorted(
        path
        for suffix in ("*.pth", "*.pth.tar", "*.pt", "*.ckpt")
        for path in root.rglob(suffix)
    )
    records = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = checkpoint_records(args.checkpoint_root)
    records = []
    for model_id, expected in MODELS.items():
        repo = args.cache_root / expected["cache"]
        remote = git(repo, "remote", "get-url", "origin")
        license_file = next(repo.glob("LICENSE*"), None)
        has_weight = any(expected["cache"].lower() in item["path"].lower() for item in checkpoints)
        records.append(
            {
                "model_id": model_id,
                **expected,
                "repo": str(repo),
                "head": git(repo, "rev-parse", "HEAD"),
                "remote": remote,
                "remote_matches_expected": normalize_remote(remote)
                == normalize_remote(expected["official_remote"]),
                "license_file": str(license_file) if license_file else None,
                "checkpoint_present": has_weight,
                "eligibility": "eligible_for_checkpoint_fetch" if not has_weight else "eligible_for_smoke",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": records, "visible_checkpoints": checkpoints}, indent=2) + "\n")


if __name__ == "__main__":
    main()
