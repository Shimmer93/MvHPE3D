#!/usr/bin/env python
"""Run SAM3DBody compact export for BEHAVE frame folders across multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", default="data/behave/frames")
    parser.add_argument("--output-root", default="data/behave/sam3dbody")
    parser.add_argument("--checkpoint-path", default="data/weights/sam3dbody_model.ckpt")
    parser.add_argument("--mhr-path", default="data/assets/mhr_model.pt")
    parser.add_argument("--detector-name", default="vitdet")
    parser.add_argument("--detector-path", default="data/weights/vitdet")
    parser.add_argument("--segmentor-name", default="")
    parser.add_argument("--segmentor-path", default="")
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--fov-path", default="")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=["k0", "k1", "k2", "k3"])
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")
    tasks = build_tasks(args)
    if not tasks:
        print("No BEHAVE SAM3DBody tasks to run.")
        return

    print(f"Queued {len(tasks)} folder jobs across GPUs {','.join(gpus)}")
    gpu_queues: dict[str, Queue[dict[str, str]]] = {gpu_id: Queue() for gpu_id in gpus}
    for task_index, task in enumerate(tasks):
        gpu_queues[gpus[task_index % len(gpus)]].put(task)

    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus) * args.workers_per_gpu) as executor:
        futures = []
        for gpu_id in gpus:
            for _ in range(args.workers_per_gpu):
                futures.append(executor.submit(run_worker, args, gpu_queues[gpu_id], gpu_id))
        for future in as_completed(futures):
            failures.extend(future.result())

    if failures:
        print(f"{len(failures)} BEHAVE SAM3DBody jobs failed.", file=sys.stderr)
        raise SystemExit(1)
    print("All BEHAVE SAM3DBody jobs finished.")


def build_tasks(args: argparse.Namespace) -> list[dict[str, str]]:
    image_root = (REPO_ROOT / args.image_root).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    requested_sequences = set(args.sequences) if args.sequences is not None else None
    tasks = []
    for sequence_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        if requested_sequences is not None and sequence_dir.name not in requested_sequences:
            continue
        for camera_id in args.cameras:
            image_folder = sequence_dir / camera_id
            if not image_folder.exists():
                print(f"[SKIP missing images] {image_folder}")
                continue
            output_folder = output_root / sequence_dir.name / camera_id
            if not args.overwrite and is_complete_output_folder(
                image_folder=image_folder,
                output_folder=output_folder,
            ):
                print(f"[SKIP complete outputs] {output_folder}")
                continue
            tasks.append(
                {
                    "sequence_id": sequence_dir.name,
                    "camera_id": camera_id,
                    "image_folder": str(image_folder),
                    "output_folder": str(output_folder),
                }
            )
    return tasks


def is_complete_output_folder(*, image_folder: Path, output_folder: Path) -> bool:
    image_paths = list(image_folder.glob("*.jpg"))
    if not image_paths:
        return False
    return all((output_folder / f"{image_path.stem}.npz").exists() for image_path in image_paths)


def run_worker(
    args: argparse.Namespace,
    task_queue: Queue[dict[str, str]],
    gpu_id: str,
) -> list[dict[str, object]]:
    failures = []
    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            return failures
        try:
            result = run_task(args, task, gpu_id)
            if result["returncode"] != 0:
                failures.append(result)
                print(
                    f"[FAIL gpu={gpu_id} code={result['returncode']}] "
                    f"{task['sequence_id']} {task['camera_id']}",
                    flush=True,
                )
            else:
                print(f"[OK gpu={gpu_id}] {task['sequence_id']} {task['camera_id']}", flush=True)
        finally:
            task_queue.task_done()


def run_task(args: argparse.Namespace, task: dict[str, str], gpu_id: str) -> dict[str, object]:
    command = [
        sys.executable,
        "external/sam-3d-body/demo_save_compact_params.py",
        "--image_folder",
        task["image_folder"],
        "--output_folder",
        task["output_folder"],
        "--checkpoint_path",
        args.checkpoint_path,
        "--mhr_path",
        args.mhr_path,
        "--detector_name",
        args.detector_name,
        "--segmentor_name",
        args.segmentor_name,
        "--fov_name",
        args.fov_name,
    ]
    if args.detector_path:
        command.extend(["--detector_path", args.detector_path])
    if args.segmentor_path:
        command.extend(["--segmentor_path", args.segmentor_path])
    if args.fov_path:
        command.extend(["--fov_path", args.fov_path])

    if args.dry_run:
        print(f"[DRY gpu={gpu_id}] {' '.join(command)}", flush=True)
        return {"task": task, "gpu": gpu_id, "returncode": 0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("MOMENTUM_ENABLED", "0")
    last_returncode = 1
    attempts = max(int(args.retries), 0) + 1
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        last_returncode = completed.returncode
        if completed.returncode == 0:
            break
        print(
            f"[RETRY gpu={gpu_id} attempt={attempt}/{attempts} code={completed.returncode}] "
            f"{task['sequence_id']} {task['camera_id']}",
            flush=True,
        )
    return {"task": task, "gpu": gpu_id, "returncode": last_returncode}


if __name__ == "__main__":
    main()
