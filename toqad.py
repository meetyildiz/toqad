# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """\
"""

_URL = "https://raw.githubusercontent.com/meetyildiz/toqad/main/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
}

class SquadConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadConfig, self).__init__(**kwargs)


class Squad(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SquadConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            
            for document in squad["data"]: 
                for par in document['paragraphs']:
                    for qas in par['qas']:
                        if len(qas['answers']) == 0: #no answer
                            ans_start = -1
                            ans_end = -1
                            ans_text = ""
                        else:
                            ans_start = int(qas['answers'][0]['answer_start'])
                            ans_end = ans_start + len(qas['answers'][0]['text'])
                            ans_text = qas['answers'][0]['text']

                        ex = {
                            "id": qas["id"],
                            "title": document["title"],
                            "context": par['context'],
                            "question": qas['question'],
                            "answers": {
                                "text": [ans_text],
                                "answer_start": [ans_start],
                            },
                        }

                        yield key, ex
                        key += 1
