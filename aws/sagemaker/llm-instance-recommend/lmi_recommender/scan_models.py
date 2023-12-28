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
import logging

from huggingface_hub import HfApi

from model_size_estimator import get_model_size_and_dtype
from model_zoo import TextGenerationModels

MIN_DOWNLOADS = 2000


def check():
    hf_models = TextGenerationModels()
    for model_id, value in hf_models.models.items():
        file_size = value["file_size"]
        if not file_size:
            logging.info(f"Invalid file_zie: {model_id}")
            continue

        dtype = value["dtype"]
        config = value["config"]
        model_info = get_model_size_and_dtype(config)
        if model_info:
            expected_dtype = model_info["dtype"]
            if dtype != expected_dtype:
                logging.info(f"{model_id}: {dtype}, inferred: {expected_dtype}")
            else:
                expected_size = model_info["size"]
                size_ratio = abs(expected_size - file_size) / (expected_size + file_size)
                if size_ratio > 0.01:
                    logging.info(f"{model_id}: model size diff: {size_ratio}")

    hf_models.save(trim=True)


def main():
    logging.info("=========== importing text-generation models ==========")

    api = HfApi()
    model_list = api.list_models(filter="text-generation-inference,gguf",
                                 sort="downloads",
                                 direction=-1,
                                 limit=None)

    hf_models = TextGenerationModels()
    total = 0
    for model in model_list:
        total += 1
        model_id = model.id
        if model.downloads < MIN_DOWNLOADS:
            logging.debug(f"Skip model {model_id}, downloads {model.downloads} < {MIN_DOWNLOADS}")
            continue

        if hf_models.contains(model_id):
            continue

        hf_models.add_model(model)
        logging.info(f"{hf_models.size()}: {model_id}.")

    hf_models.save()
    logging.info(f"Total models: {total}")
    logging.info(f"imported: {hf_models.size()}.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
