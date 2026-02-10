# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np
from pathlib import Path


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_unique_filepath(directory, filename):
    """Generate a unique filepath by adding a counter if the file already exists.

    Args:
        directory: Directory where the file should be saved
        filename: Desired filename

    Returns:
        str: Full path to a unique filename (with counter suffix if needed)
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    base_path = Path(directory) / filename
    if not base_path.exists():
        return str(base_path)

    # Split filename and extension
    name = base_path.stem
    ext = base_path.suffix

    counter = 1
    while True:
        new_path = Path(directory) / f"{name}_{counter}{ext}"
        if not new_path.exists():
            return str(new_path)
        counter += 1
