"""Geometry-only 4D-VGGT inference and reproducible prediction export."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from four_d_vggt.compat.checkpoint import load_checkpoint
from four_d_vggt.models.model import FourDVGGT
from four_d_vggt.utils.load_fn import load_and_preprocess_images
from four_d_vggt.utils.pose_enc import pose_encoding_to_extri_intri


PREDICTION_KEYS = {
    "pose_enc",
    "pose_enc_list",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "images",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(inputs):
    """Directories use natural filename order; explicit lists preserve caller order."""
    paths = [Path(path).resolve() for path in inputs]
    if len(paths) == 1 and paths[0].is_dir():
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        paths = sorted(
            (path for path in paths[0].iterdir() if path.is_file() and path.suffix.lower() in extensions),
            key=lambda path: (
                [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)],
                path.name,
            ),
        )
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Provide a nonempty image directory or an ordered list of image files")
    return paths


def load_model(checkpoint, device="cuda"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError("Checkpoint must contain a geometry-only model_config")
    model = FourDVGGT(**payload["model_config"])
    load_report = load_checkpoint(model, payload)
    model.eval().to(device)
    return model, load_report


def predict(model, images, precision="fp32", camera_decoding="all"):
    device = next(model.parameters()).device
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if camera_decoding not in {"all", "extrinsics", "none"}:
        raise ValueError("camera_decoding must be all, extrinsics or none")
    if precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 inference is supported only on CUDA; use fp32 on CPU")
    images = images.to(device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), amp:
        outputs = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    unexpected = set(outputs) - PREDICTION_KEYS
    if unexpected:
        raise ValueError(f"Model returned unsupported Version 1.0 outputs: {sorted(unexpected)}")
    arrays = {}
    for key, value in outputs.items():
        if isinstance(value, (list, tuple)):
            value = torch.stack(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unsupported prediction type for {key}: {type(value)}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite model output: {key}")
        arrays[key] = value.detach().float().cpu().numpy()
    if "pose_enc" in arrays and camera_decoding != "none":
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            torch.from_numpy(arrays["pose_enc"]),
            images.shape[-2:],
            build_intrinsics=camera_decoding == "all",
        )
        arrays["extrinsics"] = extrinsics.numpy()
        if intrinsics is not None:
            arrays["intrinsics"] = intrinsics.numpy()
    return arrays, {
        "forward_seconds": elapsed,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "precision": precision,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "camera_decoding": camera_decoding,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="4D-VGGT Version 1.0 geometry-only inference for camera, depth, and point maps"
    )
    parser.add_argument("--images", nargs="+", required=True, help="One directory or ordered image paths")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Version 1.0 geometry-only checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--preprocess", choices=("crop", "pad"), default="crop")
    parser.add_argument("--image-size", type=int, default=518, help="Multiple of 14")
    parser.add_argument("--no-camera-decoding", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    torch.manual_seed(args.seed)
    paths = image_paths(args.images)
    images = load_and_preprocess_images(paths, mode=args.preprocess, target_size=args.image_size)
    model, load_report = load_model(args.checkpoint, args.device)
    arrays, runtime = predict(
        model,
        images,
        args.precision,
        camera_decoding="none" if args.no_camera_decoding else "all",
    )
    metadata = {
        "schema_version": 1,
        "release": "version-1.0",
        "released_tasks": ["camera", "depth", "point"],
        "model": model.checkpoint_metadata,
        "model_config": model.model_config,
        "load_report": load_report,
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": args.seed,
        "images": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "input_shape": list(images.shape),
        "coordinates": "extrinsics world-to-camera",
        "runtime": runtime,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "arrays": metadata["arrays"], **runtime}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
