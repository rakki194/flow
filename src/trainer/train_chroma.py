
import sys
import os
from dataclasses import dataclass

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import vae_flatten, prepare_latent_image_ids, vae_unflatten
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params

from transformers import T5Tokenizer
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys

from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant

@dataclass
class TrainingConfig:
    master_seed: int
    accumulate_ot_batch: int

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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_chroma(rank, world_size):
    # Initialize distributed training
    setup_distributed(rank, world_size)
        
    model_config = ModelConfig(
        chroma_path = "models/flux/FLUX.1-schnell/chroma-8.9b.safetensors",
        vae_path = "models/flux/ae.safetensors",
        t5_path = "models/flux/text_encoder_2",
        t5_config_path = "models/flux/text_encoder_2/config.json",
        t5_tokenizer_path = "models/flux/tokenizer_2",
    )

    # default for debugging
    training_config = TrainingConfig(
        master_seed=0,
        accumulate_ot_batch=1024,
    )

    dataloader_config = DataloaderConfig(
        batch_size=32,
        jsonl_metadata_path="test_raw_data.jsonl",
        image_folder_path="furry_50k_4o/images",
        base_resolution=[512],
        tag_based=True,
        tag_drop_percentage=0.2,
        uncond_percentage=0.1,
        resolution_step=8,
        num_workers=2,
        prefetch_factor=2,
    )
    # global training RNG
    torch.manual_seed(training_config.master_seed)

    # load model
    with torch.no_grad():
    #     # load chroma
    #     with torch.device("meta"):
    #         model = Chroma(chroma_params)
    #     model.load_state_dict(load_safetensors(model_config.chroma_path),  assign=True)

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(replace_keys(load_file_multipart(model_config.t5_path)), assign=True)
        t5.eval()



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
        batch_size=1, # batch size is handled in the dataset
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        prefetch_factor=dataloader_config.prefetch_factor,
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )

    os.makedirs("preview", exist_ok=True)

    acc_latents = []
    acc_embeddings = []
    for i, data in tqdm(enumerate(dataloader), total=len(dataset)):
        images, caption, index = data[0]

        print(images.shape)

        # we load and unload vae and t5 here to reduce vram usage 
        # think of this as caching on the fly 
        # load t5 and vae to GPU 
        ae.to(rank)
        t5.to(rank)
        if len(acc_latents) < training_config.accumulate_ot_batch // (dataloader_config.batch_size // world_size):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # init random noise

                text_inputs = t5_tokenizer(
                    caption,
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
                print("t5_done")
                torch.cuda.empty_cache()
                latents = ae.encode_for_train(images.to(rank)).to("cpu", non_blocking=True)
                acc_latents.append(latents)
                print("vae_done")
                
                # flush
                torch.cuda.empty_cache()
        # accumulate the latents and embedding in a variable
        # unload t5 and vae
        else:
            t5.to("cpu")
            ae.to("cpu")
            torch.cuda.empty_cache()
            dist.barrier()

            # process the cache buffer now!
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ae.to(rank)
                for j, buffer in enumerate(zip(acc_embeddings, acc_latents)):
                    images = ae.decode(buffer[1].to(rank))

                    save_image(make_grid(images.clip(-1, 1)), f"preview/{j}_{i}_rank_{rank}.jpg", normalize=True)

            acc_latents = []
            acc_embeddings = []

    print()

    dist.destroy_process_group()