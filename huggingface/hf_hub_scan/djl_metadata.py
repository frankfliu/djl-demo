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
import hashlib
import json


def sha1_sum(file_name: str):
    with open(file_name, 'rb') as f:
        content = f.read()
        return hashlib.sha1(content).hexdigest()


class HuggingfaceMetadata:

    def __init__(self, model_id):
        self.artifact_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.artifacts = []

    def add_file(self, quant: str, uri: str):
        self.artifacts.append({
            "version": "0.0.1",
            "snapshot": False,
            "name": quant,
            "arguments": {
                "engine": "Llama",
                "translatorFactory":
                "ai.djl.llama.engine.LlamaTranslatorFactory"
            },
            "files": {
                "model": {
                    "uri": uri,
                    "name": f"{quant}.gguf"
                }
            }
        })

    def save_metadata(self, metadata_file: str):
        metadata = {
            "metadataVersion": "0.2",
            "resourceType": "model",
            "application": "nlp/text_generation",
            "groupId": "ai.djl.huggingface.gguf",
            "artifactId": self.artifact_id,
            "name": self.model_name,
            "description":
            f"Huggingface transformers model: {self.model_name}",
            "website": "http://www.djl.ai/engine/llama",
            "licenses": {
                "license": {
                    "name": "The Apache License, Version 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0"
                }
            },
            "artifacts": self.artifacts
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata,
                      f,
                      sort_keys=False,
                      indent=2,
                      ensure_ascii=False)
