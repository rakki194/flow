import sys
import os
from torchvision.utils import save_image, make_grid

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

from src.dataloaders.dataloader import TextImageDataset


dataset = TextImageDataset(
    batch_size=64,
    jsonl_path="test_raw_data.jsonl",
    image_folder_path="furry_50k_4o/images",
    # tag_implication_path="furry_50k_4o/tag_implications-2024-06-16.csv",
    base_res=[
        256,
        384,
        512,
    ],  # resolution computation should be in here instead of in csv prep!
    tag_based=True,
    tag_drop_percentage=0.8,
    uncond_percentage=0.1,
    resolution_step=32,
    seed=0,
    rank=0,
    num_gpus=1,
)

os.makedirs("preview", exist_ok=True)

for i in tqdm(range(10)):
    images, caption, index = dataset[i]
    save_image(make_grid(images.clip(-1, 1)), f"preview/{i}.jpg", normalize=True)

print()
