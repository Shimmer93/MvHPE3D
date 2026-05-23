#!/usr/bin/env python
"""Run SAM3DBody compact export for MPI-INF-3DHP across multiple GPUs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", default="data/mpi_inf_3dhp/frames")
    parser.add_argument("--output-root", default="data/mpi_inf_3dhp/sam3dbody")
    parser.add_argument("--checkpoint-path", default="data/weights/sam3dbody_model.ckpt")
    parser.add_argument("--mhr-path", default="data/assets/mhr_model.pt")
    parser.add_argument("--detector-name", default="vitdet")
    parser.add_argument("--detector-path", default="data/weights/vitdet")
    parser.add_argument("--segmentor-name", default="")
    parser.add_argument("--segmentor-path", default="")
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--fov-path", default="")
    parser.add_argument("--subjects", nargs="*", default=[f"S{i}" for i in range(1, 9)])
    parser.add_argument("--sequences", nargs="*", default=["Seq1", "Seq2"])
    parser.add_argument("--cameras", nargs="*", default=["video_0", "video_2", "video_7", "video_8"])
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--startup-timeout-minutes", type=int, default=20)
    parser.add_argument("--job-timeout-minutes", type=int, default=0)
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
        print("No SAM3DBody tasks to run.")
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
        print(f"{len(failures)} SAM3DBody jobs failed.", file=sys.stderr)
        raise SystemExit(1)
    print("All SAM3DBody jobs finished.")


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


def build_tasks(args: argparse.Namespace) -> list[dict[str, str]]:
    tasks = []
    image_root = (REPO_ROOT / args.image_root).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    for subject_id in args.subjects:
        for sequence_name in args.sequences:
            sequence_id = f"{subject_id}_{sequence_name}"
            for camera_id in args.cameras:
                image_folder = image_root / sequence_id / camera_id
                if not image_folder.exists():
                    print(f"[SKIP missing images] {image_folder}")
                    continue
                output_folder = output_root / sequence_id / camera_id
                if output_folder.exists() and not args.overwrite and any(output_folder.glob("*.npz")):
                    print(f"[SKIP existing outputs] {output_folder}")
                    continue
                tasks.append(
                    {
                        "sequence_id": sequence_id,
                        "camera_id": camera_id,
                        "image_folder": str(image_folder),
                        "output_folder": str(output_folder),
                    }
                )
    return tasks


def run_task(args: argparse.Namespace, task: dict[str, str], gpu_id: str) -> dict[str, object]:
    command = [
        "uv",
        "run",
        "python",
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
        "--detector_path",
        args.detector_path,
        "--segmentor_name",
        args.segmentor_name,
        "--fov_name",
        args.fov_name,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.segmentor_path:
        command.extend(["--segmentor_path", args.segmentor_path])
    if args.fov_path:
        command.extend(["--fov_path", args.fov_path])
    if args.dry_run:
        print(f"[DRY gpu={gpu_id}] {' '.join(command)}")
        return {**task, "gpu": gpu_id, "returncode": 0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("MOMENTUM_ENABLED", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_HOME", "/dysData/shimmer/.cache/torch")
    env.setdefault(
        "DINOV3_REPO_PATH",
        "/dysData/shimmer/.cache/torch/hub/facebookresearch_dinov3_main",
    )

    timeout_seconds = None
    if args.job_timeout_minutes > 0:
        timeout_seconds = args.job_timeout_minutes * 60

    attempts = max(args.retries, 0) + 1
    last_returncode = 1
    for attempt in range(1, attempts + 1):
        print(
            f"[START gpu={gpu_id} attempt={attempt}/{attempts}] "
            f"{task['sequence_id']} {task['camera_id']}",
            flush=True,
        )
        last_returncode = run_command(
            command,
            env=env,
            timeout_seconds=timeout_seconds,
            startup_timeout_seconds=(
                args.startup_timeout_minutes * 60
                if args.startup_timeout_minutes > 0
                else None
            ),
            output_folder=Path(task["output_folder"]),
        )
        if last_returncode == 0:
            return {**task, "gpu": gpu_id, "returncode": 0}
        print(
            f"[RETRY gpu={gpu_id} code={last_returncode}] "
            f"{task['sequence_id']} {task['camera_id']}",
            flush=True,
        )
    return {**task, "gpu": gpu_id, "returncode": last_returncode}


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_seconds: int | None,
    startup_timeout_seconds: int | None,
    output_folder: Path,
) -> int:
    baseline_outputs = count_npz_outputs(output_folder)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        elapsed = 0
        poll_interval = 10
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            elapsed += poll_interval
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                terminate_process_group(process)
                return 124
            if (
                startup_timeout_seconds is not None
                and elapsed >= startup_timeout_seconds
                and count_npz_outputs(output_folder) <= baseline_outputs
            ):
                terminate_process_group(process)
                return 125
            try:
                return process.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                pass
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        return 124


def count_npz_outputs(output_folder: Path) -> int:
    if not output_folder.exists():
        return 0
    return sum(1 for _ in output_folder.glob("*.npz"))


def terminate_process_group(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


if __name__ == "__main__":
    main()
