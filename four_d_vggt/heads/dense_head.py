"""Independent depth and point-map DPT prediction branches."""

from torch import nn

from .dpt_head import DPTHead


class DenseHead(nn.Module):
    def __init__(
        self,
        dim_in=2048,
        patch_size=14,
        enable_depth=True,
        enable_point=True,
        **dpt_options,
    ):
        super().__init__()
        options = dict(dim_in=dim_in, patch_size=patch_size, **dpt_options)
        self.depth_head = (
            DPTHead(**options, output_dim=2, activation="exp", conf_activation="expp1")
            if enable_depth
            else None
        )
        self.point_head = (
            DPTHead(**options, output_dim=4, activation="inv_log", conf_activation="expp1")
            if enable_point
            else None
        )

    def forward(self, tokens, images, patch_start_idx, frames_chunk_size=8):
        outputs = {}
        if self.depth_head is not None:
            outputs["depth"], outputs["depth_conf"] = self.depth_head(
                tokens, images, patch_start_idx, frames_chunk_size
            )
        if self.point_head is not None:
            outputs["world_points"], outputs["world_points_conf"] = self.point_head(
                tokens, images, patch_start_idx, frames_chunk_size
            )
        return outputs
