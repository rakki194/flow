import os
import sys
import csv
import json
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


def save_as_jsonl(data, filename):
    """Saves a list of dictionaries as a JSONL file.

    Args:
        data: A list of dictionaries.
        filename: The filename to save the JSONL file.
    """
    if os.path.join(*os.path.split(filename)[:-1]) != "":
        os.makedirs(os.path.join(*os.path.split(filename)[:-1]), exist_ok=True)

    with open(filename, "w") as f:
        for item in tqdm(data):
            json.dump(item, f)
            f.write("\n")


def read_jsonl(filename):
    """Reads a JSONL file and returns a list of dictionaries.

    Args:
      filename: The filename of the JSONL file.

    Returns:
      A list of dictionaries.
    """

    data = []
    with open(filename, "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))
    return data


def csv_to_jsonl(csv_file_path, jsonl_file_path, chunk_size=100000):
    """
    Converts a large CSV file to JSON Lines format.

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    jsonl_file_path (str): Path to the output JSON Lines file.
    chunk_size (int): Number of rows to process at a time.

    This function processes the CSV file in chunks to avoid memory issues.
    """
    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file, open(
        jsonl_file_path, mode="w", encoding="utf-8"
    ) as jsonl_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row_count, row in tqdm(enumerate(reader, start=1)):
            rows.append(row)
            # Process in chunks
            if row_count % chunk_size == 0:
                # Write chunk to JSONL
                jsonl_file.write("\n".join(json.dumps(row) for row in rows) + "\n")
                rows = []  # Clear the list to free up memory

        # Write any remaining rows after the last chunk
        if rows:
            jsonl_file.write("\n".join(json.dumps(row) for row in rows) + "\n")


def prepare_jsonl(
    jsonl_path,
    filename_col,
    caption_or_tags_col,
    width_col,
    height_col,
    ext_col=None,
    ext="",
    chunksize=1000,
):
    """
    Reads a JSONL file in chunks, processes each line, and creates a list of metadata dictionaries.

    Parameters:
    - jsonl_path (str): Path to the input JSONL file.
    - filename_col (str): Column name for the filename.
    - caption_or_tags_col (str): Column name for the caption or tags.
    - bucket_col_list (list): List of columns to include as buckets.
    - ext_col (str, optional): Column for file extension, if needed.
    - chunksize (int): Number of lines to process at a time.

    Returns:
    - list: A list of metadata dictionaries.
    """
    jsonl_pbar = tqdm(desc="Processing JSONL chunks", unit_scale=chunksize)
    jsonl = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunk = []

        for line in f:
            data = json.loads(line)
            if ext_col:
                ext = data[ext_col]
            metadata = {
                "filename": data[filename_col] + "." + ext,
                "caption_or_tags": data[caption_or_tags_col],
                "width": data[width_col],
                "height": data[height_col],
            }
            chunk.append(metadata)

            if len(chunk) >= chunksize:
                jsonl.extend(chunk)
                chunk = []
                jsonl_pbar.update(1)

        # Add any remaining lines after the last chunk
        if chunk:
            jsonl.extend(chunk)
            jsonl_pbar.update(1)

    return jsonl
