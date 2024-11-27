from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

from .module.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    _use_compiled: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            use_compiled=params._use_compiled,
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img.to(next(self.img_in.parameters()).device))
        vec = self.time_in(timestep_embedding(timesteps, 256).to(self.time_in.device))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(self.guidance_in.device)).to(vec.device)
        vec = vec + self.vector_in(y.to(self.vector_in.device)).to(vec.device)
        txt = self.txt_in(txt.to(next(self.txt_in.parameters()).device))

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            # just in case in different GPU for simple pipeline parallel
            double_block_device = block.device
            if self.training:
                img, txt = ckpt.checkpoint(
                    block,
                    img.to(double_block_device),
                    txt.to(double_block_device),
                    vec.to(double_block_device),
                    pe.to(double_block_device),
                )
            else:
                img, txt = block(
                    img=img.to(double_block_device),
                    txt=txt.to(double_block_device),
                    vec=vec.to(double_block_device),
                    pe=pe.to(double_block_device),
                )

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_block_device = block.device
            if self.training:
                img = ckpt.checkpoint(
                    block,
                    img.to(single_block_device),
                    vec.to(single_block_device),
                    pe.to(single_block_device),
                )
            else:
                img = block(
                    img.to(single_block_device),
                    vec=vec.to(single_block_device),
                    pe=pe.to(single_block_device),
                )
        img = img[:, txt.shape[1] :, ...]

        final_layer_device = self.final_layer.device
        img = self.final_layer(
            img.to(final_layer_device), vec.to(final_layer_device)
        )  # (N, T, patch_size ** 2 * out_channels)
        return img
