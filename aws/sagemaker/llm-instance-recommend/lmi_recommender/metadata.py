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

from huggingface_hub.hf_api import ModelInfo

from model_size_estimator import calculate_model_size


class ModelMetadata:

    def __init__(self, model_info: ModelInfo, config: dict):
        self.model_info = model_info
        self.config = config
        self.architecture = None
        self.model_type = None
        self.dtype = "float16"
        self.size = -1
        self.sha1 = model_info.sha
        self.downloads = model_info.downloads
        self.gguf = None
        self._set_model_info()

    def _set_model_info(self):
        if "architectures" in self.config and self.config["architectures"]:
            self.architecture = self.config["architectures"][0]

        if "model_type" in self.config:
            self.model_type = self.config["model_type"]
        elif "llama" in self.model_info.tags:
            self.model_type = "llama"
        else:
            self.model_type = "unknown"

        if "torch_dtype" in self.config:
            self.dtype = self.config["torch_dtype"]
        else:
            self.dtype = "float32"

        model_id = self.model_info.id
        self.size, self.gguf = calculate_model_size(model_id)

    def get_metadata(self, with_config: bool = True):
        ret = {
            "architecture": self.architecture,
            "model_type": self.model_type,
            "dtype": self.dtype,
            "file_size": self.size,
            "sha1": self.sha1,
            "downloads": self.downloads,
        }
        if with_config:
            ret["config"] = self.config
        if self.gguf:
            ret["gguf"] = self.gguf

        return ret
