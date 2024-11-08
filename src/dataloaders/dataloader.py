import hashlib
import os
import ast
import csv
import math
import random
import logging
from collections import defaultdict
import color_profile_handling
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset

import json
from tqdm import tqdm
from PIL import Image

from bucketing_logic import create_bucket_column

log = logging.getLogger(__name__)


def _prune(tags: str, tree):
    # Filter nodes with no child nodes in the same set
    filtered_tags = [
        node for node in tags if not any(child in tags for child in tree[node])
    ]
    # random.shuffle(filtered_tags)
    # filtered_tags = sorted(filtered_tags)
    return filtered_tags


def _sample_elements_by_percentage(my_list, percentage):
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")

    sample_size = math.ceil(len(my_list) * percentage)
    return random.sample(my_list, sample_size)


def _create_tree(csv_path):
    tree = defaultdict(list)
    with open(csv_path, "rt") as csvfile:
        reader = csv.DictReader(csvfile)
        # next(reader) # id,antecedent_name,consequent_name,created_at,status
        for row in reader:
            if row["status"] == "active":
                tree[row["consequent_name"]].append(row["antecedent_name"])
    return tree


def _load_and_process_data(batch_size, csv_path,):
    bucketed_posts = {}

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        # load and group the bucket into groups
        for row in tqdm(reader, desc="Loading data"):
            if row["bucket"] is not None:
                sample = {
                    # NOTE: this is bad extract the column names out as variables!
                    "md5": row["md5"],
                    "id": int(row["id"]),
                    "tags": row["tag_string"],
                    "captions": row["caption"],
                    "file_ext": row["file_ext"],
                    "bucket": ast.literal_eval(row["bucket"]),
                }

                # fallback to md5.ext if there's no filename col
                if "filename" in row:
                    sample["filename"] = row["filename"]
                else:
                    sample["filename"] = row["md5"] + "." + row["file_ext"]

                if row["bucket"] in bucketed_posts:
                    bucketed_posts[row["bucket"]].append(sample)
                else:
                    bucketed_posts[row["bucket"]] = [sample]

    # pre-shuffle bucket for maximum randomness :P
    for key in tqdm(bucketed_posts.keys(), desc="shuffling buckets"):
        random.shuffle(bucketed_posts[key])
    # simple post counter
    post_count = 0
    for k in bucketed_posts:
        post_count += len(bucketed_posts[k])
    log.info(f"There are {post_count} text image pairs.")

    # arrange into batches
    batches = []
    for b in bucketed_posts:
        samples = []
        # if not full batch drop the last batch to prevent bad things
        for s in bucketed_posts[b]:
            if len(samples) < batch_size:
                samples.append(s)
            elif len(samples) == batch_size:
                batches.append(samples)
                samples = [s]
    log.info(f"We got {len(batches)} batches from these pairs.")

    # Theops: The batches are already shuffled here all along.
    random.shuffle(batches)

    # calculate the hash of this version of shuffled data
    # so we know which cached version is it when resuming
    return batches, bucketed_posts


def scale_and_crop_long_axis(image: Image, target_height, target_width):
    # resize image to the standard size of this bucket
    if (target_width / target_height) >= (image.width / image.height):
        image = v2.functional.resize(
            image,
            [round(target_width * image.height / image.width), target_width],
            interpolation=v2.InterpolationMode.LANCZOS,
        )
        image = v2.functional.center_crop(
            image,
            [target_height, target_width],
        )
    else:
        image = v2.functional.resize(
            image,
            [target_height, round(image.width * target_height / image.height)],
            interpolation=v2.InterpolationMode.LANCZOS,
        )
        image = v2.functional.center_crop(
            image,
            [target_height, target_width],
        )
    return image


class E6Dataset(Dataset):
    def __init__(
        self,
        batch_size,
        csv_path,
        implication_csv_path,
        image_folder_path,
        base_res=[1024],
        res_increment=8,
        max_aspect_ratio=2,
        height_col="image_height",
        width_col="image_width",
        cache_dir="cache",
        tag_drop_percentage=0.8,
        uncond_percentage=0.01,
        seed=0,
        rank=0,
        num_gpus=1,
        resume_batch_pointer=0,
    ):
        # coarsened dataset, the batch is handled by the dataset and not the dataloader,
        # increase dataloader prefetch so this thing  run optimally!
        random.seed(seed)
        # create cache folder
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # just simple pil image to tensor conversion
        self.image_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize([0.5], [0.5]),
            ]
        )

        # Theops: create them from scratch
        # put this variable here for now, don't want to expose it
        _e6_dump_with_bucket_csv = os.path.join(cache_dir, "bucket_metadata.csv")
        _bucket_col = "bucket"

        # append the bucket to the csv and dump the bucket csv in cache folder
        create_bucket_column(
            in_csv_path=csv_path,
            out_csv_path=_e6_dump_with_bucket_csv,
            base_resolution=base_res,
            step=res_increment,
            ratio_cutoff=max_aspect_ratio,
            height_col_name=height_col,
            width_col_name=width_col,
            bucket_col_name=_bucket_col,
            return_bucket=False,
        )
        # load the csv as grouped bucket dictionary
        self.metadata, _ = _load_and_process_data(batch_size, _e6_dump_with_bucket_csv)

        self.hash = self._calc_str_hash(json.dumps(self.metadata))

        if resume_batch_pointer is not None:
            self.metadata = self.metadata[resume_batch_pointer:]


        # e621 tag tree
        self.tag_implication_tree = _create_tree(implication_csv_path)

        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.tag_drop_percentage = tag_drop_percentage
        self.uncond_percentage = uncond_percentage
        self.num_gpus = num_gpus
        self.rank = rank

        # slice metadata using round robbin
        self._round_robin()

    @staticmethod
    def dummy_collate_fn(batch):
        return batch

    def _calc_str_hash(self, s):
        str_sha256 = hashlib.sha256(s.encode('utf-8')).hexdigest()
        return str(str_sha256)


    def get_hash(self,):
        return self.hash


    def _round_robin(self):
        # reason we do round robbin here instead of classic torch distributed batch is because
        # we have bucketing and the shape is different for each gpu
        # drop last batch
        self.metadata = self.metadata[:len(self.metadata) - len(self.metadata) % self.num_gpus]
        # slice round-robin
        subset_for_this_worker = []
        for i in range(0, len(self.metadata), self.num_gpus):
            subset_for_this_worker.append(self.metadata[i + self.rank])
            # print(i + self.rank)
        self.metadata = subset_for_this_worker

    def __len__(self):
        return len(self.metadata)

    def get_prompt(self, tags, caption):
        tmp = random.random()
        if tmp >= 1-self.uncond_percentage:
            # Theops: dropout
            return ""
        elif caption == "":
            # Theops: don't have a caption yet
            # so it is tags or dropout
            return tags
        else:
            # Theops: we have captions.
            # Now we have five options:
            # tags
            # tags + caption
            # caption + tags
            # only caption
            # dropout
            # Let's assign probabilities to them.
            # Note that all of these are for debugging, and they by no means indicates the optimal
            # values that I recommend.
            # p(tags) = 0.1
            # p(tags + caption) = 0.2
            # p(caption + tags) = 0.2
            # p(caption) = 0.4
            # p(dropout) = 0.1

            if tmp < 0.1:
                # tags only
                return tags
            elif 0.1 <= tmp < 0.3:
                # tags + caption
                return tags + "\n" + caption
            elif 0.3 <= tmp < 0.5:
                # caption + tags
                return caption + "\n" + tags
            elif 0.5 <= tmp < 0.9:
                # only caption
                return caption

    def get_full_prompt(self, tags, caption):
        tmp = random.random()
        if tmp >= 1-self.uncond_percentage:
            # Theops: dropout
            return ""
        else:
            return "image tags: " + tags + " image description: " + caption

    def get_batches(self):
        return self.metadata


    def __getitem__(self, index):
        batch = self.metadata[index]
        images = []
        training_prompts = []

        for i, sample in enumerate(batch):
            try:
                # just for testing echoing
                # if i == 2:
                #     raise Exception("test incomplete batch!")

                # image processing
                standard_width, standard_height = sample["bucket"]
                # prioritize jxl over other format
                image_path = os.path.join(self.image_folder_path, sample["md5"]) + ".jxl"
                if os.path.exists(image_path):
                    image = color_profile_handling.open_srgb(image_path).convert("RGB")
                else:
                    image_path = os.path.join(self.image_folder_path, sample["filename"])
                    image = Image.open(image_path).convert("RGB")
                # crop to bucket size
                image = scale_and_crop_long_axis(image, standard_height, standard_width)
                image = self.image_transforms(image)
                images.append(image)

            except Exception as e:
                log.error(f"An error occurred: {e} for {sample['filename']}")
                continue

            # tags processing
            tags = sample["tags"].split(
                " "
            )  # TODO: filter this to only use species tags and position
            random.shuffle(tags)
            tags = _prune(tags, self.tag_implication_tree)
            tags = _sample_elements_by_percentage(
                tags, random.uniform(1 - self.tag_drop_percentage, 1)
            )
            # coma separated
            tags = [s.replace('_', ' ') for s in tags]
            tags = ", ".join(tags)

            # captions processing
            # captions.append(sample["captions"])

            # Theops: training prompt processing
            # training_prompt = self.get_prompt(tags, sample["captions"])
            training_prompt = self.get_full_prompt(tags, sample["captions"])
            training_prompts.append(training_prompt)

        # echo short batch
        while len(images) < 1:
            log.info(
                f"An Empty batch is caught! This batch will be discarded and a random batch will be fetch."
            )
            new_batch_index = random.randrange(0, len(self.metadata))
            return self.__getitem__(new_batch_index)

        images = torch.stack(images, dim=0)

        # Theops: we now return index, so we will be able to save it when stopping
        return images, training_prompts, index


if __name__ == "__main__":

    # testing round robbin simulating multiple gpu training
    data = E6Dataset(
        10,
        "flux-pilot-dataset-1024/pilot_cluster.csv",
        "flux-pilot-dataset-1024/tag_implications-2024-08-05.csv",
        "flux-pilot-dataset-1024/images",
        [1024],
        64,
        2,
        tag_drop_percentage=0.0,
        uncond_percentage=0.9,
        num_gpus=7,
        rank=0
    )

    p = data[0]
    # q = data2[0]
    # r = data3[0]
    print()
