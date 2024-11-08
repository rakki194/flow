import csv
import sys
import math
import random
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


def _bucket_generator(base_resolution=256, ratio_cutoff=2, step=64):
    # bucketing follows the logic of 1/x
    # https://www.desmos.com/calculator/byizruhsry
    x = base_resolution
    y = base_resolution
    half_buckets = [(x, y)]
    while y / x <= ratio_cutoff:
        y += step
        x = int((base_resolution**2 / y) // step * step)
        if x != half_buckets[-1][0]:
            half_buckets.append(
                (
                    x,
                    y,
                )
            )

    another_half_buckets = []
    for bucket in reversed(half_buckets[1:]):
        another_half_buckets.append(bucket[::-1])

    return another_half_buckets + half_buckets


def _euclidian_distance_2d(x1, y1, x2, y2):
    result = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return result


def _normalize_width_height(w, h):
    # scale the width and height to a unit value
    t = math.sqrt(1.0 / (w * h))
    w = w * t
    h = h * t
    return w, h


def _closest_bucket(w, h, standarized_bucket_dict):
    w, h = _normalize_width_height(w, h)

    # loop over the bucket and compute the closest 2d euclidian distance to the bucket
    # this ensures the closest bucket and not exceeding the threshold
    ration_err = []  # store the error for all possible bucket
    for b_st in standarized_bucket_dict.keys():
        ration_err.append(
            (_euclidian_distance_2d(*b_st, w, h), standarized_bucket_dict[b_st])
        )
    # loop through and find the shortest distance
    min_err, closest_b = ration_err[0]
    for err, b in ration_err:
        if err < min_err:
            min_err = err
            closest_b = b

    return closest_b


def create_bucket_column(
    in_csv_path,
    out_csv_path,
    base_resolution, # Need to be an array like this: `[384, 512, 640, 768, 896, 1024]`
    step=8,
    ratio_cutoff=2,
    height_col_name="image_height",
    width_col_name="image_width",
    bucket_col_name="bucket",
    return_bucket=False,
):
    # this may explode if the csv is ridiculously big
    all_rows = []

    with open(in_csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in tqdm(reader, desc="loading csv"):
            all_rows.append(row)

    # image resolution metadata columns
    image_height_index = header.index(height_col_name)
    image_width_index = header.index(width_col_name)
    multires_buckets = []
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        multires_buckets.append(buckets)

    # bucket scaling factor normalized
    # for example (1, 1) meaning it's square scaled
    # (1.41, 0.70) <- (1448, 720) non square normalization
    standardized_buckets = list()
    for buckets in tqdm(multires_buckets, total=len(multires_buckets), desc="creating multi-res buckets"):
        this_standardized_bucket = dict()
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)  # (w, h)
            this_standardized_bucket[b_st] = bucket
        standardized_buckets.append(this_standardized_bucket)

    # iterate csv rows
    for row in tqdm(all_rows, total=len(all_rows), desc="appending bucket"):
        # grab image width and height
        image_width = int(row[image_width_index])
        image_height = int(row[image_height_index])

        # guard check
        if image_width > 0 and image_height > 0:
            aspect_ratio = image_width/image_height
            if ratio_cutoff > aspect_ratio > 1./ratio_cutoff:
                w, h = _normalize_width_height(image_width, image_height)

                # randomly choose a resolution
                this_standardized_bucket = random.choice(standardized_buckets)

                closest_b = _closest_bucket(w, h, this_standardized_bucket)
                row.append(str(closest_b))
            else:
                pass
        else:
            pass

    # save it
    with open(out_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header.append(bucket_col_name)
        writer.writerow(header)
        for row in all_rows:
            writer.writerow(row)

    if return_bucket:
        return standardized_buckets
