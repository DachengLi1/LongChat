# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from longchat.conversation import get_default_conv_template, SeparatorStyle
from tqdm import tqdm

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    #num_data: int = field(
    #    default=-1, metadata={"help": "Number of training data to use."}
    #)
    begin: int = field(
        default=-1, metadata={"help": "Begin Index"}
    )
    min_lr_ratio: float = field(
        default=0.5, metadata={"help": "Minimal Learning rate"}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, begin: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        #self.stages = {"8192": 500000, "16384": 500000, "32768": 300000, "65536": 100000}
        self.stages = {"8192": 500000}#, "16384":1000 ,"32768": 152500, "65536": 5000}
        self.total_num = self.stages[str(self.max_length)]

        rank0_print("Loading data...")
        with open(data_path, 'r') as json_file:
            list_data_dict = list(json_file)
        
        print(len(list_data_dict))
        if begin != -1:
            rank0_print("Starting from {begin} out of {len(list_data_dict)}")
            list_data_dict = list_data_dict[begin:]
        self.begin = begin
        assert (list(self.stages.keys()).index(str(self.max_length)) == 0 or begin != -1), "You have to start in the middle if this is not the first stage!"
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        print(f"Initializeing distributed dataset: {self.rank} / {self.world_size}")
        self.prefetch_num = 100
        self.cache = {"input_ids": [], "labels": [], "attention_mask": []}
        self.dataset_index = 0
#        self._prefetch()
    
    def __len__(self):
        return self.total_num

    def _prefetch(self):
        orig_index = self.dataset_index
        for i in tqdm(range(self.prefetch_num)):
            cur_data = json.loads(self.list_data_dict[i + self.dataset_index])
            cur_text = cur_data["text"]
            cur_tokenized_text = self.tokenizer(cur_text, return_tensors="pt")

            cur_len = len(cur_tokenized_text.input_ids[0])
            # Drop the remainder
            cur_len = (cur_len // self.max_length) * self.max_length
            # Split text into block of size self.max_length    
            split = {
                k: [t[0][j : j + self.max_length] for j in range(0, cur_len, self.max_length)]
                for k, t in cur_tokenized_text.items()
            }
            split["labels"] = split["input_ids"].copy()
            
            for k, t in split.items():
                self.cache[k].extend(t)
            self.dataset_index += 1
        print(f"Prefetched the next {self.prefetch_num} data from {orig_index}, now at {self.dataset_index + self.begin} , with {len(self.cache['input_ids'])} cache data")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # This is a hack: disregard the actual index..
        num_per_rank = len(i) if isinstance(i, list) else 1
        num_total = num_per_rank * self.world_size
        start_id = num_per_rank * self.rank
        end_id = start_id + num_per_rank

        while len(self.cache["input_ids"]) < num_total:
            self._prefetch()
            print(f"Rank {self.rank} fetching {start_id} to {end_id}, first 10 elements: {self.cache['input_ids'][start_id][:10]}")
        
        data_dict = dict(
                input_ids=self.cache["input_ids"][start_id:end_id],
                labels=self.cache["labels"][start_id:end_id],
                attention_mask=self.cache["attention_mask"][start_id:end_id]
                )

        # Update cache
        self.cache["input_ids"] = self.cache["input_ids"][num_total:]
        self.cache["labels"] = self.cache["labels"][num_total:]
        self.cache["attention_mask"] = self.cache["attention_mask"][num_total:]
       
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                attention_mask=data_dict["attention_mask"][0],
            )
        return data_dict


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset
    )
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, begin=data_args.begin)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def pretrain():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
   # from ..monkey_patch.llama_bias_monkey_patch import LlamaForCausalLMBias
   # model = LlamaForCausalLMBias.from_pretrained(
   #     model_args.model_name_or_path,
   #     cache_dir=training_args.cache_dir,
   # )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    print(training_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
