import os
import json

def save_as_jsonl(data, filename):
    """Saves a list of dictionaries as a JSONL file.

    Args:
        data: A list of dictionaries.
        filename: The filename to save the JSONL file.
    """
    if os.path.join(*os.path.split(filename)[:-1]) != "":
        os.makedirs(os.path.join(*os.path.split(filename)[:-1]), exist_ok=True)
  
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def read_jsonl(filename):
  """Reads a JSONL file and returns a list of dictionaries.

  Args:
    filename: The filename of the JSONL file.

  Returns:
    A list of dictionaries.
  """

  data = []
  with open(filename, 'r') as f:
    for line in f:
      data.append(json.loads(line))
  return data
