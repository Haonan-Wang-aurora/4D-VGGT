"""Strict loader for Version 1.0 geometry-only checkpoints."""

from collections.abc import Mapping

import torch

from four_d_vggt.metadata import ARCHITECTURE_SCHEMA_VERSION, GEOMETRY_PROFILE


def load_checkpoint(model, payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint model state is empty or malformed")
    if payload.get("architecture") != GEOMETRY_PROFILE:
        raise ValueError("Checkpoint is not a Version 1.0 geometry-only checkpoint")
    if payload.get("architecture_schema_version") != ARCHITECTURE_SCHEMA_VERSION:
        raise ValueError("Missing or unsupported architecture schema version")
    if payload.get("model_config") != model.model_config:
        raise ValueError("Checkpoint model_config must exactly match the constructed model")
    if set(state) != set(model.state_dict()):
        missing = sorted(set(model.state_dict()) - set(state))[:12]
        unexpected = sorted(set(state) - set(model.state_dict()))[:12]
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(state, strict=True)
    return {
        "mode": "geometry_v1_strict",
        "loaded_tensors": len(state),
        "released_tasks": ["camera", "depth", "point"],
    }


def checkpoint_payload(model):
    if hasattr(model, "module"):
        model = model.module
    return {
        **model.checkpoint_metadata,
        "model_config": model.model_config,
        "model": model.state_dict(),
    }
