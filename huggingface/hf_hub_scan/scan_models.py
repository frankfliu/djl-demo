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
import argparse
import logging
import os
import shutil
import sys

from huggingface_hub import HfApi

from djl_metadata import HuggingfaceMetadata
from imported_model import ImportedModels
from model_size_estimator import get_model_size_and_dtype


def save_djl_model_zoo(category: str):
    hf_models = ImportedModels(category)

    for model_id, value in hf_models.models.items():
        temp_dir = "temp"
        gguf = value.get("gguf")
        if not gguf:
            continue

        repo_dir = f"output/model/nlp/text_generation/ai/djl/huggingface/gguf/{model_id}"
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)

        metadata = HuggingfaceMetadata(model_id)

        for quant, gguf_file in gguf.items():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            file_name = gguf_file["name"]

            # Save serving.properties
            serving_file = os.path.join(temp_dir, "serving.properties")
            with open(serving_file, 'w') as f:
                f.write(
                    f"engine=Llama\n"
                    f"option.modelName={file_name[:-5]}\n"
                    f"translatorFactory=ai.djl.llama.engine.LlamaTranslatorFactory\n"
                )

            uri = f"https://huggingface.co/{model_id}/resolve/main/{file_name}?download=true"
            metadata.add_file(quant, uri)

        metadata_file = os.path.join(repo_dir, "metadata.json")
        metadata.save_metadata(metadata_file)


def trim(category: str):
    hf_models = ImportedModels(category)
    for model_id, value in hf_models.models.items():
        file_size = value["file_size"]
        if not file_size:
            if "gguf" not in value:
                logging.info(f"Invalid file_zie: {model_id}")
            continue

        dtype = value["dtype"]
        config = value["config"]
        model_info = get_model_size_and_dtype(config)
        if model_info:
            expected_dtype = model_info["dtype"]
            if dtype != expected_dtype:
                logging.info(
                    f"{model_id}: {dtype}, inferred: {expected_dtype}")
            else:
                expected_size = model_info["size"]
                size_ratio = abs(expected_size - file_size) / (expected_size +
                                                               file_size)
                if size_ratio > 0.01:
                    logging.info(f"{model_id}: model size diff: {size_ratio}")

    hf_models.save(trim=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--trim-metadata",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="convert full category json file to compact version")
    parser.add_argument("-g",
                        "--gguf",
                        type=bool,
                        action=argparse.BooleanOptionalAction,
                        help="import gguf models from huggingface")
    parser.add_argument("-c",
                        "--model-category",
                        type=str,
                        required=False,
                        default="text-generation",
                        help="model category")
    parser.add_argument("-m",
                        "--min-downloads",
                        type=int,
                        required=False,
                        default=2000,
                        help="minimum monthly downloads")
    parser.add_argument("-u",
                        "--update-metadata",
                        type=bool,
                        action=argparse.BooleanOptionalAction,
                        help="Force update existing model metadata")

    args = parser.parse_args()

    if args.trim_metadata:
        trim(args.model_category)
        return

    if args.gguf:
        save_djl_model_zoo(args.model_category)
        return

    logging.info(
        f"=========== importing {args.model_category} models ==========")

    api = HfApi()
    model_list = api.list_models(filter=args.model_category,
                                 sort="downloads",
                                 direction=-1,
                                 limit=None)
    model_list = [
        model for model in model_list if 'text-generation' in model.tags
        or 'text-generation-inference' in model.tags
    ]

    hf_models = ImportedModels(args.model_category)
    total = 0
    new_models = 0
    for model in model_list:
        total += 1
        model_id = model.id
        if model.downloads < args.min_downloads:
            logging.debug(
                f"Skip model {model_id}, downloads {model.downloads} < {args.min_downloads}"
            )
            continue

        if hf_models.contains(model_id):
            if args.update_metadata:
                m = hf_models.get_model_info(model_id)
                m["downloads"] = model.downloads
            continue

        new_models += 1
        logging.info(f"Scanning {new_models}: {model_id}.")

        hf_models.add_model(model)
        if new_models % 50 == 0:
            hf_models.save()

    hf_models.save()
    logging.info(f"Total models on huggingface hub: {total}")
    logging.info(f"imported: {new_models} of {hf_models.size()}.")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    main()
