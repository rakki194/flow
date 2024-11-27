from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

from .module.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    SingleStreamBlock,
    timestep_embedding,
    Approximator,
    distribute_modulations,
)


@dataclass
class ChromaParams:
    in_channels: int
    # vec_in_dim: int
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
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    _use_compiled: bool


chroma_params = (
    ChromaParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
        approximator_in_dim=64,
        approximator_depth=5,
        approximator_hidden_size=5120,
        _use_compiled=False,
    ),
)


class Chroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: ChromaParams):
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

        # TODO: need proper mapping for this approximator output!
        # currently the mapping is hardcoded in distribute_modulations function
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
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

        # TODO: move this hardcoded value to config
        self.mod_index_length = 344
        self.mod_index = torch.tensor(list(range(self.mod_index_length)))

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # hardcoded value

        distill_timestep = timestep_embedding(timesteps, 16)
        # TODO: need to add toggle to omit this from schnell but that's not a priority
        distil_guidance = timestep_embedding(guidance, 16)
        # get all modulation index
        modulation_index = timestep_embedding(self.mod_index, 32)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = (
            torch.cat([distill_timestep, distil_guidance], dim=1)
            .unsqueeze(1)
            .repeat(1, self.mod_index_length, 1)
        )
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
        mod_vectors = self.distilled_guidance_layer(input_vec)
        mod_vectors_dict = distribute_modulations(mod_vectors)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            # the guidance replaced by FFN output
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]

            # just in case in different GPU for simple pipeline parallel
            if self.training:
                img, txt = ckpt.checkpoint(block, img, txt, pe, double_mod)
            else:
                img, txt = block(img=img, txt=txt, pe=pe, distill_vec=double_mod)

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if self.training:
                img = ckpt.checkpoint(block, img, pe, single_mod)
            else:
                img = block(img, pe=pe, distill_vec=single_mod)
        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(
            img, distill_vec=final_mod
        )  # (N, T, patch_size ** 2 * out_channels)
        return img
