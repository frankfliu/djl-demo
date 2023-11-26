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
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import GatedRepoError

from metadata import ModelMetadata


class TextGenerationModels:

    def __init__(self):
        self.api = HfApi()
        self.models = {}

        file = "model_catalog_full.json"
        if os.path.exists(file):
            with open(file, "r") as f:
                self.models = json.load(f)

    def items(self):
        return self.models.items()

    def get_model_info(self, model_id: str) -> Optional[dict]:
        return self.models.get(model_id)

    def contains(self, model_id: str) -> bool:
        return model_id in self.models

    def save(self, trim: bool = False):
        file_name = "model_catalog_full.json"
        if trim:
            file_name = "model_catalog.json"
            for _, model in self.models.items():
                model.pop("config")
                model.pop("downloads")

        with open(file_name, 'w') as f:
            json.dump(self.models,
                      f,
                      sort_keys=False,
                      indent=2,
                      ensure_ascii=False)

    def add_model(self, model_info: ModelInfo):
        model_id = model_info.id
        hf_token = os.environ.get("HF_TOKEN", None)
        try:
            config_file = hf_hub_download(repo_id=model_id, filename="config.json", token=hf_token)
        except GatedRepoError:
            logging.warning(f"Access denied for model: {model_id}, please provide access token.")
            return

        with open(config_file) as f:
            config = json.loads(f.read())

        metadata = ModelMetadata(model_info, config)
        self.models[model_id] = metadata.get_metadata()

    def size(self) -> int:
        return len(self.models)
