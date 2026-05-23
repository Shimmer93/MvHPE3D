#!/usr/bin/env python
"""Run MPI-INF-3DHP GT-skeleton SMPL fitting shards across multiple GPUs."""

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
    parser.add_argument("--dataset-root", default="/dysData/shimmer/datasets/mpi_inf_3dhp")
    parser.add_argument("--manifest-path", default="data/mpi_inf_3dhp/mpi_inf_3dhp_stage2_manifest.json")
    parser.add_argument("--output-dir", default="data/mpi_inf_3dhp/gt_smpl_fit")
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--annotation-key", choices=("univ_annot3", "annot3"), default="univ_annot3")
    parser.add_argument("--splits", nargs="*", default=("train", "val"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--fit-space", choices=("world", "input_root"), default="world")
    parser.add_argument("--input-smpl-cache-dir", default="data/mpi_inf_3dhp/sam3dbody_fitted_smpl")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument(
        "--fit-joint-source",
        choices=("smpl24", "smpl24_headtop_proxy", "regressor", "regressor_headtop_proxy"),
        default="smpl24_headtop_proxy",
    )
    parser.add_argument("--joint-regressor-path", default=None)
    parser.add_argument("--headtop-topk", type=int, default=32)
    parser.add_argument("--headtop-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--pose-reg", type=float, default=0.02)
    parser.add_argument("--shape-reg", type=float, default=0.1)
    parser.add_argument("--bone-len-reg", type=float, default=0.01)
    parser.add_argument("--temporal-reg", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
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
    print(f"Launching {num_shards} SMPL-fitting shards across GPUs {','.join(gpus)}")

    failures = []
    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = []
        for shard_index in range(num_shards):
            futures.append(
                executor.submit(
                    run_shard,
                    args=args,
                    gpu_id=gpus[shard_index % len(gpus)],
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
                print(f"[OK gpu={result['gpu']} shard={result['shard_index']}]", flush=True)

    if failures:
        print(f"{len(failures)} SMPL-fitting shards failed.", file=sys.stderr)
        raise SystemExit(1)
    print("All SMPL-fitting shards finished.")


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
        "scripts/fit_mpi_inf_3dhp_gt_smpl.py",
        "--dataset-root",
        args.dataset_root,
        "--manifest-path",
        args.manifest_path,
        "--output-dir",
        args.output_dir,
        "--smpl-model-path",
        args.smpl_model_path,
        "--annotation-key",
        args.annotation_key,
        "--fit-space",
        args.fit_space,
        "--input-smpl-cache-dir",
        args.input_smpl_cache_dir,
        "--num-views",
        str(args.num_views),
        "--device",
        "cuda:0",
        "--batch-size",
        str(args.batch_size),
        "--num-iters",
        str(args.num_iters),
        "--lr",
        str(args.lr),
        "--fit-joint-source",
        args.fit_joint_source,
        "--headtop-topk",
        str(args.headtop_topk),
        "--headtop-axis",
        args.headtop_axis,
        "--joint-weight",
        str(args.joint_weight),
        "--pose-reg",
        str(args.pose_reg),
        "--shape-reg",
        str(args.shape_reg),
        "--bone-len-reg",
        str(args.bone_len_reg),
        "--temporal-reg",
        str(args.temporal_reg),
        "--grad-clip",
        str(args.grad_clip),
        "--progress-every",
        str(args.progress_every),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if args.splits:
        command.append("--splits")
        command.extend(args.splits)
    if args.sequences:
        command.append("--sequences")
        command.extend(args.sequences)
    if args.cameras:
        command.append("--cameras")
        command.extend(args.cameras)
    if args.joint_regressor_path is not None:
        command.extend(["--joint-regressor-path", args.joint_regressor_path])
    if args.skip_existing:
        command.append("--skip-existing")
    else:
        command.append("--no-skip-existing")
    if args.overwrite:
        command.append("--overwrite")
    if args.max_frames_per_sequence is not None:
        command.extend(["--max-frames-per-sequence", str(args.max_frames_per_sequence)])

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
