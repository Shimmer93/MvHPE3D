#!/usr/bin/env python
"""Precompute frozen RGB image features for multiview manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.rgb_features import resolve_rgb_feature_cache_path  # noqa: E402
from mvhpe3d.data.splits import load_sample_records  # noqa: E402
from mvhpe3d.visualization.smpl_overlay import resolve_rgb_image_path  # noqa: E402

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen timm RGB encoder and cache one feature vector per view."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Path to a Stage 1/2/3 manifest JSON file.",
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default=None,
        help="Directory containing cropped RGB images. Defaults to <manifest_dir>/rgb.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="RGB feature cache directory. Defaults to <manifest_dir>/rgb_features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_small_patch16_224.dino",
        help="timm model name for the frozen encoder.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize the timm model without pretrained weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Square input resolution for the frozen encoder. Defaults to the "
            "model's pretrained input size when available."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Encoder batch size."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.image_size is not None and args.image_size < 1:
        raise ValueError(f"--image-size must be >= 1, got {args.image_size}")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError(f"--max-images must be >= 1, got {args.max_images}")

    manifest_path = Path(args.manifest_path).resolve()
    manifest_dir = manifest_path.parent
    rgb_dir = Path(args.rgb_dir).resolve() if args.rgb_dir else manifest_dir / "rgb"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else manifest_dir / "rgb_features"
    )
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(manifest_path)
    items = collect_unique_view_items(records)
    if args.max_images is not None:
        items = items[: args.max_images]
    if not items:
        print("No manifest views found. Nothing to precompute.")
        return

    device = resolve_device(args.device)
    model = build_frozen_timm_encoder(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        device=device,
    )
    image_size = resolve_encoder_image_size(model, requested_image_size=args.image_size)

    pending_items: list[tuple[str, str, str, Path, Path]] = []
    skipped = 0
    for sequence_id, camera_id, frame_id in items:
        cache_path = resolve_rgb_feature_cache_path(
            output_dir,
            sequence_id=sequence_id,
            camera_id=camera_id,
            frame_id=frame_id,
        )
        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue
        image_path = resolve_rgb_image_path(
            rgb_dir,
            sequence_id=sequence_id,
            camera_id=camera_id,
            frame_id=frame_id,
        )
        pending_items.append((sequence_id, camera_id, frame_id, image_path, cache_path))

    processed = 0
    for batch in tqdm(chunked(pending_items, args.batch_size), desc="RGB features"):
        if not batch:
            continue
        images = [
            load_and_preprocess_image(image_path, image_size=image_size)
            for _sequence_id, _camera_id, _frame_id, image_path, _cache_path in batch
        ]
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).to(
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            features = normalize_feature_output(model(image_tensor))
        features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)

        for item, feature in zip(batch, features_np, strict=True):
            _sequence_id, _camera_id, _frame_id, image_path, cache_path = item
            save_rgb_feature_cache(
                cache_path,
                rgb_feature=feature,
                encoder_name=args.model_name,
                image_path=image_path,
                image_size=image_size,
            )
            processed += 1

    print(f"Manifest: {manifest_path}")
    print(f"RGB dir: {rgb_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Encoder: {args.model_name}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    print(f"Views: {len(items)}")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def build_frozen_timm_encoder(
    *,
    model_name: str,
    pretrained: bool,
    device: torch.device,
) -> torch.nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def resolve_encoder_image_size(
    model: torch.nn.Module,
    *,
    requested_image_size: int | None,
) -> int:
    expected_size = infer_encoder_image_size(model)
    if requested_image_size is None:
        return expected_size
    if requested_image_size != expected_size and has_fixed_patch_embed_size(model):
        raise ValueError(
            f"The selected encoder expects {expected_size}x{expected_size} inputs, "
            f"but --image-size was set to {requested_image_size}. Use "
            f"--image-size {expected_size}, or omit --image-size to use the model default."
        )
    return requested_image_size


def infer_encoder_image_size(model: torch.nn.Module) -> int:
    patch_embed = getattr(model, "patch_embed", None)
    img_size = getattr(patch_embed, "img_size", None)
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        height = int(img_size[0])
        width = int(img_size[1])
        if height == width and height > 0:
            return height
    if isinstance(img_size, int) and img_size > 0:
        return int(img_size)

    pretrained_cfg = getattr(model, "pretrained_cfg", None) or getattr(
        model,
        "default_cfg",
        None,
    )
    if isinstance(pretrained_cfg, dict):
        input_size = pretrained_cfg.get("input_size")
        if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
            height = int(input_size[-2])
            width = int(input_size[-1])
            if height == width and height > 0:
                return height

    return 224


def has_fixed_patch_embed_size(model: torch.nn.Module) -> bool:
    patch_embed = getattr(model, "patch_embed", None)
    return getattr(patch_embed, "img_size", None) is not None


def collect_unique_view_items(records) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    items: list[tuple[str, str, str]] = []
    for record in records:
        for view in record.views:
            item = (str(record.sequence_id), str(view.camera_id), str(record.frame_id))
            if item in seen:
                continue
            seen.add(item)
            items.append(item)
    return sorted(items)


def load_and_preprocess_image(image_path: Path, *, image_size: int) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def normalize_feature_output(features: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError(f"Expected encoder output tensor, got {type(features)!r}")
    if features.ndim == 2:
        return features.float()
    if features.ndim == 3:
        return features[:, 0].float()
    if features.ndim == 4:
        return features.mean(dim=(2, 3)).float()
    raise ValueError(f"Unsupported encoder output shape: {tuple(features.shape)}")


def save_rgb_feature_cache(
    cache_path: Path,
    *,
    rgb_feature: np.ndarray,
    encoder_name: str,
    image_path: Path,
    image_size: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        rgb_feature=np.asarray(rgb_feature, dtype=np.float32),
        encoder_name=np.asarray(encoder_name),
        image_path=np.asarray(str(image_path)),
        image_size=np.asarray([image_size, image_size], dtype=np.int32),
        valid=np.asarray(True),
    )


def chunked(items: list, chunk_size: int) -> list[list]:
    return [
        items[index : index + chunk_size] for index in range(0, len(items), chunk_size)
    ]


if __name__ == "__main__":
    main()
