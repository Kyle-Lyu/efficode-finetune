import gc
import copy
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

import torch
import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


IGNORE_TOKEN_ID = -100


@dataclass
class DataCollatorWithDynamicPad:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable value is "pt".
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_TOKEN_ID
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.return_tensors == "pt", 'Only support Torch framework, please set `return_tensors="pt"`'

        # The inputs' length can't exceed `max_length`
        for feature in features:
            for key in feature.keys():
                feature[key] = feature[key][:self.max_length]
        
        labels = [feature["labels"] for feature in features]
        no_labels_features = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        
        if padding_side == "right":
            batch["labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        return batch
    

@dataclass
class DataCollatorWithDynamicPack:
    """
    Data collator that will dynamically pack the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable value is "pt".
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_TOKEN_ID
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.return_tensors == "pt", 'Only support Torch framework, please set `return_tensors="pt"`'

        # The inputs' length can't exceed `max_length`
        for feature in features:
            for key in feature.keys():
                feature[key] = feature[key][:self.max_length]

        # Sort features by input_ids length
        features.sort(key=lambda x: len(x["input_ids"]))

        batched_features = list()
        packed_sample = {
            "input_ids": list(),
            "attention_mask": list(),
            "labels": list(),
        }
        for feature in features:
            if (len(packed_sample["input_ids"]) + len(feature["input_ids"])) > self.max_length:
                batched_features.append(copy.deepcopy(packed_sample))
                packed_sample.clear()
                packed_sample["input_ids"] = list()
                packed_sample["attention_mask"] = list()
                packed_sample["labels"] = list()

            packed_sample["input_ids"].extend(feature["input_ids"])
            packed_sample["attention_mask"].extend(feature["attention_mask"])
            packed_sample["labels"].extend(feature["labels"])
        batched_features.append(copy.deepcopy(packed_sample))

        labels = [feature["labels"] for feature in batched_features]
        no_labels_features = [{k: v for k, v in feature.items() if k != "labels"} for feature in batched_features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        
        if padding_side == "right":
            batch["labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        return batch
    

# Converting Bytes to Gigabytes
def b2gb(x):
    return x / 2**30


class MemoryUsageCallback(TrainerCallback):
    def __init__(self, local_rank, device):
        self.local_rank = local_rank
        self.device = device
        self.max_memory = 0
        self.total_memory = 0
        self.count = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        gc.collect()
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        # Reset the "peak" stats tracked by the CUDA memory allocator.
        torch.cuda.reset_peak_memory_stats()
        
        # Return the current GPU memory occupied by tensors in bytes for a given device.
        self.tensor_begin = torch.cuda.memory_allocated()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Memory occupied by tensors before entering the train: {b2gb(self.tensor_begin):.2f} GB")

        # Return the current GPU memory managed by the caching allocator in bytes for a given device.
        self.cache_begin = torch.cuda.memory_reserved()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Memory occupied by the caching allocator before entering the train: {b2gb(self.cache_begin):.2f} GB")

    # def on_step_end(self, args, state, control, **kwargs):
    #     torch.cuda.empty_cache()
    #     gc.collect()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tensor_end = torch.cuda.memory_allocated()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Memory occupied by tensors consumed at the end of the train: {b2gb(self.tensor_end):.2f} GB")
        
        # Return the maximum GPU memory occupied by tensors in bytes for a given device.
        self.tensor_peak = torch.cuda.max_memory_allocated()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Peak Memory occupied by tensors consumed during the train: {b2gb(self.tensor_peak):.2f} GB")
        
        self.cache_end = torch.cuda.memory_reserved()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Memory occupied by the caching allocator consumed at the end of the train: {b2gb(self.cache_end):.2f} GB")
        
        # Return the maximum GPU memory managed by the caching allocator in bytes for a given device.
        self.cache_peak = torch.cuda.max_memory_reserved()
        print(f"[Process rank: {self.local_rank}, device: {self.device}] GPU Peak Memory occupied by the caching allocator consumed during the train: {b2gb(self.cache_peak):.2f} GB")
        # print(f"[Process rank: {self.local_rank}, device: {self.device}] Total GPU Peak Memory consumed during the train: {b2gb(self.cache_peak):.2f} GB")

        torch.cuda.empty_cache()
        gc.collect()


CHAT_TEMPLATE = '''{% if not add_generation_prompt is defined %}
{% set add_generation_prompt = false %}
{% endif %}
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{{bos_token}}{%- if not ns.found -%}
{{'You are a helpful programming assistant.\n'}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{ message['content'] }}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'### Instruction:\n' + message['content'] + '\n'}}
        {%- else %}
{{'### Response:\n' + message['content']}}{{eos_token}}{{'\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}
{{'### Response:'}}
{% endif %}'''

SEPARATOR = "### Response:\n"
