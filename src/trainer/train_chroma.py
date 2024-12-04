import sys
import os
from dataclasses import dataclass

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from torchastic import Compass, StochasticAccumulator

from transformers import T5Tokenizer

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant


@dataclass
class TrainingConfig:
    master_seed: int
    cache_minibatch: int
    train_minibatch: int
    trained_layer_keywords: list[str]
    offload_param_count: int


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    tag_based: bool
    tag_drop_percentage: float
    uncond_percentage: float
    resolution_step: int
    num_workers: int
    prefetch_factor: int


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_sot_pairings(latents):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    image_pos_id = prepare_latent_image_ids(n, h, w)

    # randomize ode timesteps
    input_timestep = torch.round(
        F.sigmoid(torch.randn((n,), device=latents.device)), decimals=3
    )
    timesteps = input_timestep[:, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = latents * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    target = noise - latents

    return noisy_latents, target, input_timestep, image_pos_id, latent_shape


def train_chroma(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    model_config = ModelConfig(
        chroma_path="models/flux/FLUX.1-schnell/chroma-8.9b.safetensors",
        vae_path="models/flux/ae.safetensors",
        t5_path="models/flux/text_encoder_2",
        t5_config_path="models/flux/text_encoder_2/config.json",
        t5_tokenizer_path="models/flux/tokenizer_2",
    )

    # default for debugging
    training_config = TrainingConfig(
        master_seed=0,
        cache_minibatch=2,
        train_minibatch=1,
        offload_param_count=6000000000,
        trained_layer_keywords=[f"double_blocks.{x}" for x in range(14, 15)]
        + [f"single_blocks.{x}" for x in range(6, 8)],
    )

    dataloader_config = DataloaderConfig(
        batch_size=8,
        jsonl_metadata_path="test_raw_data.jsonl",
        image_folder_path="furry_50k_4o/images",
        base_resolution=[512],
        tag_based=True,
        tag_drop_percentage=0.2,
        uncond_percentage=0.1,
        resolution_step=64,
        num_workers=2,
        prefetch_factor=2,
    )
    # global training RNG
    torch.manual_seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        with torch.device("meta"):
            model = Chroma(chroma_params)
        model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)

        for name, param in model.named_parameters():
            if any(
                keyword in name for keyword in training_config.trained_layer_keywords
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False  # Optionally disable grad for others
        StochasticAccumulator.assign_hooks(model)

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(
            replace_keys(load_file_multipart(model_config.t5_path)), assign=True
        )
        t5.eval()
        t5.to(torch.bfloat16)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        # don't use this tag implication pruning it's slow!
        # preprocess the jsonl tags before training!
        # tag_implication_path="tag_implications.csv",
        base_res=dataloader_config.base_resolution,
        tag_based=dataloader_config.tag_based,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # batch size is handled in the dataset
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        prefetch_factor=dataloader_config.prefetch_factor,
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )

    os.makedirs("preview", exist_ok=True)

    for counter, data in tqdm(enumerate(dataloader), total=len(dataset)):
        images, caption, index = data[0]

        print(images.shape)

        # we load and unload vae and t5 here to reduce vram usage
        # think of this as caching on the fly
        # load t5 and vae to GPU
        ae.to(rank)
        t5.to(rank)

        acc_latents = []
        acc_embeddings = []
        for mb_i in tqdm(
            range(
                dataloader_config.batch_size
                // training_config.cache_minibatch
                // world_size
            )
        ):
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                # init random noise
                text_inputs = t5_tokenizer(
                    caption[
                        mb_i
                        * training_config.cache_minibatch : mb_i
                        * training_config.cache_minibatch
                        + training_config.cache_minibatch
                    ],
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                ).to(t5.device)

                # offload to cpu
                t5_embed = t5(text_inputs.input_ids).to("cpu", non_blocking=True)
                acc_embeddings.append(t5_embed)

                # flush
                torch.cuda.empty_cache()
                latents = ae.encode_for_train(
                    images[
                        mb_i
                        * training_config.cache_minibatch : mb_i
                        * training_config.cache_minibatch
                        + training_config.cache_minibatch
                    ].to(rank)
                ).to("cpu", non_blocking=True)
                acc_latents.append(latents)

                # flush
                torch.cuda.empty_cache()

        # accumulate the latents and embedding in a variable
        # unload t5 and vae

        t5.to("cpu")
        ae.to("cpu")
        torch.cuda.empty_cache()
        if not debug:
            dist.barrier()

        # move model to device
        model.to(rank)

        acc_latents = torch.cat(acc_latents, dim=0)
        acc_embeddings = torch.cat(acc_embeddings, dim=0)

        # process the cache buffer now!
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # prepare flat image and the target lerp
            (
                noisy_latents,
                target,
                input_timestep,
                image_pos_id,
                latent_shape,
            ) = prepare_sot_pairings(acc_latents.to(rank))
            noisy_latents = noisy_latents.to(torch.bfloat16)
            target = target.to(torch.bfloat16)
            input_timestep = input_timestep.to(torch.bfloat16)
            image_pos_id = image_pos_id.to(rank)

            # t5 text id for the model
            text_ids = torch.zeros((noisy_latents.shape[0], 512, 3), device=rank)
            # NOTE:
            # using static guidance 1 for now
            # this should be disabled later on !
            static_guidance = torch.tensor([1.0] * acc_latents.shape[0], device=rank)

        # set the input to requires grad to make autograd works
        noisy_latents.requires_grad_(True)
        acc_embeddings.requires_grad_(True)

        ot_bs = acc_latents.shape[0]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # preview_images = []
            for tmb_i in tqdm(
                range(
                    dataloader_config.batch_size
                    // training_config.train_minibatch
                    // world_size
                )
            ):
                # do this inside for loops!

                # NOTE: doing this one batch at a time but we can change it later
                pred = model(
                    img=noisy_latents[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                    img_ids=image_pos_id[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                    txt=acc_embeddings[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ].to(rank),
                    txt_ids=text_ids[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                    timesteps=input_timestep[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                    guidance=static_guidance[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                )
                loss = F.mse_loss(
                    pred,
                    target[
                        tmb_i
                        * training_config.train_minibatch : tmb_i
                        * training_config.train_minibatch
                        + training_config.train_minibatch
                    ],
                )
                loss.backward()

                # check if the batching is correct

                # with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                #     # debug preview
                #     ae.to(rank)
                #     preview_images.append(
                #         ae.decode(
                #             vae_unflatten(
                #                 noisy_latents[tmb_i * training_config.train_minibatch: tmb_i * training_config.train_minibatch + training_config.train_minibatch],
                #                 latent_shape
                #             )
                #         )
                #     )

            # preview_images = torch.cat(preview_images)
            # save_image(make_grid(preview_images.clip(-1, 1)), f"preview_noisy/{counter}_{tmb_i}_rank_{rank}.jpg", normalize=True)

        # offload some params to cpu just enough to make room for the caching process
        # and only offload non trainable params
        offload_param_count = 0
        for name, param in model.named_parameters():
            if not any(
                keyword in name for keyword in training_config.trained_layer_keywords
            ):
                if offload_param_count < training_config.offload_param_count:
                    offload_param_count += param.numel()
                    param.data = param.data.to("cpu")

        images = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # debug preview
            ae.to(rank)
            for latent in tqdm(acc_latents):
                images.append(ae.decode(latent.unsqueeze(0).to(rank)))

        images = torch.cat(images)
        save_image(
            make_grid(images.clip(-1, 1)),
            f"preview/{counter}_rank_{rank}.jpg",
            normalize=True,
        )

        acc_latents = []
        acc_embeddings = []

    print()

    if not debug:
        dist.destroy_process_group()
