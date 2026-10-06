# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Geometry-only 4D-VGGT model released in Version 1.0."""

import json
from collections.abc import Mapping, Sequence

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from four_d_vggt.heads.dense_head import DenseHead
from four_d_vggt.heads.spatial_head import SpatialHead
from four_d_vggt.heads.spatiotemporal_dense_fusion import SpatiotemporalDenseFusion
from four_d_vggt.metadata import ARCHITECTURE_SCHEMA_VERSION, GEOMETRY_PROFILE, checkpoint_metadata
from four_d_vggt.models.cross_time_local_fusion import CrossTimeLocalFusion, branch_parameter_report
from four_d_vggt.models.cross_view_global_fusion import CrossViewGlobalFusion


def _plain_config(value):
    if isinstance(value, Mapping):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config(item) for item in value]
    return value


class FourDVGGT(nn.Module, PyTorchModelHubMixin):
    """Predict camera parameters, depth maps, and world-space point maps."""

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_spatial=True,
        enable_point=True,
        enable_depth=True,
        profile=GEOMETRY_PROFILE,
        enable_cross_time=True,
        enable_dense_fusion=True,
        window_size=5,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        patch_embed="dinov2_vitl14_reg",
        cached_layer_indices=(4, 11, 17, 23),
        aa_block_size=1,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        spatial_head_options=None,
        dense_head_options=None,
    ):
        super().__init__()
        if profile != GEOMETRY_PROFILE:
            raise ValueError(f"Version 1.0 supports only the geometry profile: {GEOMETRY_PROFILE}")
        if enable_dense_fusion and not enable_cross_time:
            raise ValueError("Dense fusion requires Cross-Time features")
        cached_layer_indices = tuple(cached_layer_indices)
        if (
            len(cached_layer_indices) != 4
            or len(set(cached_layer_indices)) != 4
            or min(cached_layer_indices) < 0
            or max(cached_layer_indices) >= depth
        ):
            raise ValueError("Specify four distinct valid cached_layer_indices for the configured depth")
        spatial_head_options = dict(spatial_head_options or {})
        dense_head_options = dict(dense_head_options or {})
        dense_head_options.setdefault("intermediate_layer_idx", list(cached_layer_indices))
        if tuple(dense_head_options["intermediate_layer_idx"]) != cached_layer_indices:
            raise ValueError("Dense Head intermediate indices must equal fusion cached indices")

        configuration = {
            key: value
            for key, value in locals().copy().items()
            if key not in {"self", "__class__"}
        }
        self.model_config = json.loads(json.dumps(_plain_config(configuration)))
        self.profile = profile
        self.checkpoint_metadata = {
            **checkpoint_metadata(),
            "architecture": profile,
            "architecture_schema_version": ARCHITECTURE_SCHEMA_VERSION,
            "released_tasks": ["camera", "depth", "point"],
        }
        self.model_name = self.checkpoint_metadata["model_name"]

        self.cross_view_global_fusion = CrossViewGlobalFusion(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            patch_embed=patch_embed,
            cached_layer_indices=cached_layer_indices,
            aa_block_size=aa_block_size,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            rope_freq=rope_freq,
            init_values=init_values,
        )
        self.cross_time_local_fusion = (
            CrossTimeLocalFusion(self.cross_view_global_fusion, window_size) if enable_cross_time else None
        )
        self.spatiotemporal_dense_fusion = (
            SpatiotemporalDenseFusion(2 * embed_dim, cached_layer_indices) if enable_dense_fusion else None
        )
        self.spatial_head = SpatialHead(dim_in=2 * embed_dim, **spatial_head_options) if enable_spatial else None
        self.dense_head = (
            DenseHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                enable_depth=enable_depth,
                enable_point=enable_point,
                **dense_head_options,
            )
            if enable_depth or enable_point
            else None
        )
        self.parameter_report = (
            branch_parameter_report(self.cross_view_global_fusion, self.cross_time_local_fusion)
            if enable_cross_time
            else None
        )

    def forward(self, images: torch.Tensor):
        """Run geometry inference on ``[S,3,H,W]`` or ``[B,S,3,H,W]`` images."""
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape) < 1:
            raise ValueError("images must be nonempty [S,3,H,W] or [B,S,3,H,W]")
        if not images.is_floating_point() or any(
            size % self.model_config["patch_size"] for size in images.shape[-2:]
        ):
            raise ValueError("images must be floating RGB, with H and W divisible by patch_size")

        encoded = self.cross_view_global_fusion.encode_images(images)
        spatial, patch_start_idx = self.cross_view_global_fusion.forward_tokens(encoded)
        temporal = self.cross_time_local_fusion(encoded)[0] if self.cross_time_local_fusion is not None else None
        dense_features = (
            self.spatiotemporal_dense_fusion(spatial, temporal)
            if self.spatiotemporal_dense_fusion is not None
            else spatial
        )
        predictions = {}
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.spatial_head is not None:
                poses = self.spatial_head(spatial)
                predictions.update(pose_enc=poses[-1], pose_enc_list=poses)
            if self.dense_head is not None:
                predictions.update(self.dense_head(dense_features, images, patch_start_idx))
        if not self.training:
            predictions["images"] = images
        return predictions
