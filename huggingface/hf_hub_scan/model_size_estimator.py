#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import json
import logging
import os
import re
import subprocess
from typing import List, Optional


def load_known_model_size():
    with open("known_models.json", "r") as f:
        return json.load(f)


KNOWN_MODELS = load_known_model_size()


def match_identifier(identifier: dict, candidate: dict) -> bool:
    for key, value in identifier.items():
        if key not in candidate or value != candidate[key]:
            return False
    return True


def get_model_size_and_dtype(config: dict) -> Optional[dict]:
    for key, value in KNOWN_MODELS.items():
        if match_identifier(value["identifier"], config):
            return value["model_info"]

    return None


def size_collector(lines: List[str], target='.safetensors'):
    # TODO: https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator
    sizes = []
    for line in lines:
        if target in line:
            result = line[line.find("(") + 1:line.find(")")].split()
            size, label = float(result[0]), result[1]
            if label == "MB":
                size /= 1024
            elif label == "KB" or label == "B":
                size = 0.001
            elif label != "GB":
                logging.info(f"Invalid line: {line}")

            sizes.append(size)

    return sizes


def collect_gguf_info(lines: List[str]):
    gguf = {}
    for line in lines:
        if ".gguf" not in line:
            continue

        file_name = line.split()[2]
        if not file_name.endswith(".gguf"):
            continue

        name_part = file_name[:-5]
        matcher = re.search("[.\-](q\\d)_?([\\dk])_?([sml])?(-parts-\d\d)?",
                            name_part, re.I)
        if not matcher:
            logging.warning(f"quant method not found: {file_name}")
            continue

        if matcher.group(4):
            logging.warning(f"partitioned gguf is not support: {file_name}")
            continue

        quant = f"{matcher.group(1)}_{matcher.group(2)}"
        if matcher.group(3):
            quant = f"{quant}_{matcher.group(3)}"
        quant = quant.upper()
        result = line[line.find("(") + 1:line.find(")")].split()
        size, label = float(result[0]), result[1]
        if label == "MB":
            size /= 1024
        elif label == "KB" or label == "B":
            size = 0.001
        elif label != "GB":
            logging.info(f"Invalid line: {line}")
        gguf[quant] = {"name": file_name, "size": size}

    return gguf if len(gguf) > 0 else None


def calculate_model_size(model_id: str):
    if os.environ.get("HF_TOKEN", None):
        url = f"git@hf.co:{model_id}"
    else:
        url = f"https://huggingface.co/{model_id}"
    subprocess.run(["git", "clone", "-q", "--no-checkout", url, 'test'])
    result = os.popen("cd test && git lfs ls-files -s").read()
    subprocess.run(["rm", "-rf", "test"])
    result = result.split('\n')
    gguf = collect_gguf_info(result)
    sizes = size_collector(result)
    if not sizes:
        sizes = size_collector(result, '.bin')
    final_sizes = sum(sizes)
    if gguf is None and final_sizes == 0:
        logging.info(f"Failed to calculate model size: {url}")
        logging.info(f"======== {result}")
        return None, None
    return final_sizes, gguf
