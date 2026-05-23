#!/usr/bin/env python
"""Run input SMPL precompute shards across multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--smpl-model-path", default=None)
    parser.add_argument("--mhr-assets-dir", default=None)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")

    num_shards = len(gpus) * args.workers_per_gpu
    print(f"Launching {num_shards} precompute shards across GPUs {','.join(gpus)}")
    failures = []
    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = []
        for shard_index in range(num_shards):
            gpu_id = gpus[shard_index % len(gpus)]
            futures.append(
                executor.submit(
                    run_shard,
                    args=args,
                    gpu_id=gpu_id,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)
                print(
                    f"[FAIL gpu={result['gpu']} shard={result['shard_index']} "
                    f"code={result['returncode']}]",
                    flush=True,
                )
            else:
                print(
                    f"[OK gpu={result['gpu']} shard={result['shard_index']}]",
                    flush=True,
                )

    if failures:
        print(f"{len(failures)} precompute shards failed.", file=sys.stderr)
        raise SystemExit(1)
    print("All precompute shards finished.")


def run_shard(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, int | str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/precompute_input_smpl.py",
        "--manifest-path",
        args.manifest_path,
        "--device",
        "cuda:0",
        "--batch-size",
        str(args.batch_size),
        "--progress-every",
        str(args.progress_every),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if args.cache_dir is not None:
        command.extend(["--cache-dir", args.cache_dir])
    if args.smpl_model_path is not None:
        command.extend(["--smpl-model-path", args.smpl_model_path])
    if args.mhr_assets_dir is not None:
        command.extend(["--mhr-assets-dir", args.mhr_assets_dir])
    if args.skip_existing:
        command.append("--skip-existing")

    if args.dry_run:
        print(f"[DRY gpu={gpu_id} shard={shard_index}] {' '.join(command)}", flush=True)
        return {"gpu": gpu_id, "shard_index": shard_index, "returncode": 0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return {
        "gpu": gpu_id,
        "shard_index": shard_index,
        "returncode": completed.returncode,
    }


if __name__ == "__main__":
    main()
