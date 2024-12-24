import sys
import os
from dataclasses import dataclass

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from torchastic import Compass, StochasticAccumulator
import random

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
    offload_param_count: int
    lr: float
    weight_decay: float
    warmup_steps: int
    change_layer_every: int
    trained_single_blocks: int
    trained_double_blocks: int
    save_every: int
    save_folder: str


@dataclass
class InferenceConfig:
    inference_every: int
    inference_folder: str
    steps: int
    guidance: int
    cfg: int
    prompts: list[str]
    first_n_steps_wo_cfg: int
    image_dim: tuple[int, int]
    t5_max_length: int


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
    t5_to_8bit: bool


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


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)
    # init optimizer
    optimizer = Compass(
        [
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" not in name and "norm" not in name)
                ]
            },
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" in name or "norm" in name)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, hooks, trained_params


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def optimizer_state_to(optimizer, device):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            # Check if the item is a tensor
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_part(model, trained_layer_keywords, counter, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(
        filtered_state_dict, os.path.join(save_folder, f"trained_part_{counter}.pth")
    )


def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)


def inference_wrapper(
    model,
    ae,
    t5_tokenizer,
    t5,
    seed: int,
    steps: int,
    guidance: int,
    cfg: int,
    prompts: list,
    rank: int,
    first_n_steps_wo_cfg: int,
    image_dim=(512, 512),
    t5_max_length=512,
):
    #############################################################################
    # test inference
    # aliasing
    SEED = seed
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    GUIDANCE = guidance
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg
    DEVICE = model.device
    PROMPT = prompts

    T5_MAX_LENGTH = t5_max_length

    # store device state of each model
    t5_device = t5.device
    ae_device = ae.device
    model_device = model.device
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # init random noise
            noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, SEED)
            noise, shape = vae_flatten(noise)
            noise = noise.to(rank)
            n, c, h, w = shape
            image_pos_id = prepare_latent_image_ids(n, h, w).to(rank)

            timesteps = get_schedule(STEPS, noise.shape[1])

            model.to("cpu")
            ae.to("cpu")
            t5.to(rank)  # load t5 to gpu
            text_inputs = t5_tokenizer(
                PROMPT,
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)

            t5_embed = t5(text_inputs.input_ids).to(rank)
            text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)

            ae.to("cpu")
            t5.to("cpu")
            model.to(rank)  # load model to gpu
            latent_cfg = denoise_cfg(
                model,
                noise,
                image_pos_id,
                t5_embed,
                torch.zeros_like(t5_embed),
                text_ids,
                timesteps,
                GUIDANCE,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
            )

            model.to("cpu")
            t5.to("cpu")
            ae.to(rank)  # load ae to gpu
            output_image = ae.decode(vae_unflatten(latent_cfg, shape))

            # restore back state
            model.to("cpu")
            t5.to("cpu")
            ae.to("cpu")

    # grid = make_grid(output_image, nrow=2, padding=2, normalize=True)

    return output_image


def train_chroma(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    ### ALL CONFIG IS HERE! ###
    model_config = ModelConfig(
        chroma_path="models/flux/FLUX.1-schnell/chroma-8.9b.safetensors",
        vae_path="models/flux/ae.safetensors",
        t5_path="models/flux/text_encoder_2",
        t5_config_path="models/flux/text_encoder_2/config.json",
        t5_tokenizer_path="models/flux/tokenizer_2",
        t5_to_8bit=True,
    )

    # default for debugging
    training_config = TrainingConfig(
        master_seed=0,
        cache_minibatch=2,
        train_minibatch=1,
        offload_param_count=5000000000,
        trained_single_blocks=4,
        trained_double_blocks=4,
        weight_decay=0.0001,
        lr=1e-5,
        warmup_steps=1,
        change_layer_every=3,
        save_every=6,
        save_folder="testing",
    )

    dataloader_config = DataloaderConfig(
        batch_size=8,
        jsonl_metadata_path="test_raw_data.jsonl",
        image_folder_path="furry_50k_4o/images",
        base_resolution=[256],
        tag_based=True,
        tag_drop_percentage=0.2,
        uncond_percentage=0.1,
        resolution_step=64,
        num_workers=2,
        prefetch_factor=2,
    )

    inference_config = InferenceConfig(
        inference_every=2,
        inference_folder="inference_folder",
        steps=20,
        guidance=3,
        first_n_steps_wo_cfg=-1,
        image_dim=(512, 512),
        t5_max_length=512,
        cfg=1,
        prompts=[
            "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
            # "baked potato, on the space floating orbiting around the earth",
        ],
    )

    os.makedirs(inference_config.inference_folder, exist_ok=True)
    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        with torch.device("meta"):
            model = Chroma(chroma_params)
        model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)

        # randomly train inner layers at a time
        trained_double_blocks = list(range(len(model.double_blocks)))
        trained_single_blocks = list(range(len(model.single_blocks)))
        random.shuffle(trained_double_blocks)
        random.shuffle(trained_single_blocks)
        # lazy :P
        trained_double_blocks = trained_double_blocks * 1000000
        trained_single_blocks = trained_single_blocks * 1000000

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
        if model_config.t5_to_8bit:
            cast_linear(t5, torch.float8_e4m3fn)

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

    optimizer = None
    scheduler = None
    hooks = []
    optimizer_counter = 0
    for counter, data in tqdm(
        enumerate(dataloader),
        total=len(dataset),
        desc=f"training, Rank {rank}",
        position=rank,
    ):
        if counter % training_config.change_layer_every == 0:
            # periodically remove the optimizer and swap it with new one

            # aliasing to make it cleaner
            o_c = optimizer_counter
            n_ls = training_config.trained_single_blocks
            n_ld = training_config.trained_double_blocks
            trained_layer_keywords = (
                [
                    f"double_blocks.{x}."
                    for x in trained_double_blocks[o_c * n_ld : o_c * n_ld + n_ld]
                ]
                + [
                    f"single_blocks.{x}."
                    for x in trained_single_blocks[o_c * n_ls : o_c * n_ls + n_ls]
                ]
                + ["txt_in", "img_in", "final_layer"]
            )

            # remove hooks and load the new hooks
            if len(hooks) != 0:
                hooks = [hook.remove() for hook in hooks]

            optimizer, scheduler, hooks, trained_params = init_optimizer(
                model,
                trained_layer_keywords,
                training_config.lr,
                training_config.weight_decay,
                training_config.warmup_steps,
            )

            optimizer_counter += 1

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
            ),
            desc=f"preparing latents, Rank {rank}",
            position=rank,
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

        # aliasing
        mb = training_config.train_minibatch
        for tmb_i in tqdm(
            range(dataloader_config.batch_size // mb // world_size),
            desc=f"minibatch training, Rank {rank}",
            position=rank,
        ):
            # do this inside for loops!
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
                    img=noisy_latents[tmb_i * mb : tmb_i * mb + mb],
                    img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb],
                    txt=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(rank),
                    txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb],
                    timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb],
                    guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb],
                )
                # TODO: need to scale the loss with rank count and grad accum!
                loss = F.mse_loss(
                    pred,
                    target[tmb_i * mb : tmb_i * mb + mb],
                )
            loss.backward()

        # offload some params to cpu just enough to make room for the caching process
        # and only offload non trainable params
        offload_param_count = 0
        for name, param in model.named_parameters():
            if not any(keyword in name for keyword in trained_layer_keywords):
                if offload_param_count < training_config.offload_param_count:
                    offload_param_count += param.numel()
                    param.data = param.data.to("cpu")
        torch.cuda.empty_cache()

        StochasticAccumulator.reassign_grad_buffer(model)

        if not debug:
            synchronize_gradients(model)

        scheduler.step()
        optimizer_state_to(optimizer, rank)
        optimizer.step()
        optimizer.zero_grad()

        optimizer_state_to(optimizer, "cpu")
        torch.cuda.empty_cache()

        if (counter + 1) % training_config.save_every == 0 and rank == 0:
            save_part(
                model, trained_layer_keywords, counter, training_config.save_folder
            )
        if not debug:
            dist.barrier()

        if (counter + 1) % inference_config.inference_every == 0:

            images_tensor = inference_wrapper(
                model=model,
                ae=ae,
                t5_tokenizer=t5_tokenizer,
                t5=t5,
                seed=training_config.master_seed + rank,
                steps=inference_config.steps,
                guidance=inference_config.guidance,
                cfg=inference_config.cfg,
                prompts=inference_config.prompts,
                rank=rank,
                first_n_steps_wo_cfg=inference_config.first_n_steps_wo_cfg,
                image_dim=inference_config.image_dim,
                t5_max_length=inference_config.t5_max_length,
            )
            # gather from all gpus
            if not debug:
                gather_list = (
                    [torch.empty_like(images_tensor) for _ in range(world_size)]
                    if rank == 0
                    else None
                )
                dist.gather(images_tensor, gather_list=gather_list, dst=0)
            if rank == 0:
                # Concatenate gathered tensors
                if not debug:
                    gathered_images = torch.cat(
                        gather_list, dim=0
                    )  # (total_images, C, H, W)
                else:
                    gathered_images = images_tensor
                # Create a grid
                grid = make_grid(
                    gathered_images, nrow=8, normalize=True
                )  # Adjust nrow as needed

                # Save the grid
                file_path = os.path.join(
                    inference_config.inference_folder, f"{counter}.png"
                )
                save_image(grid, file_path)
                print(f"Image grid saved to {file_path}")

        # flush
        acc_latents = []
        acc_embeddings = []

    print()

    if not debug:
        dist.destroy_process_group()
