import logging
import math
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.utils import check_min_version
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

from utils import IGNORE_TOKEN_ID, CHAT_TEMPLATE, SEPARATOR, DataCollatorWithDynamicPad, DataCollatorWithDynamicPack, MemoryUsageCallback


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.47.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        },
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            "help": "Quantization type `fp4` or `nf4`",
            "choices": ["fp4", "nf4"],
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "The path to the data."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_length: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum lenght of training samples."}
    )
    pad_mode: Optional[str] = field(
        default="pad", 
        metadata={
            "help": "Padding mode for preprocess the data.",
            "choices": ["pad", "pack"],
        },
    )


def prompt2inputs(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    split_word: str,
    label_pad_token_id: int=IGNORE_TOKEN_ID,
):
    input_ids = []
    label_ids = []
    attention_mask = []

    eos_token = tokenizer.eos_token
    # Split prompt into multiple chat rounds
    rounds = prompt.split(eos_token)
    for cur_round in rounds:
        # Skip the invalid chat round
        if len(cur_round.strip()) == 0:
            continue
        
        # In the current round, the first part is the instruction while the second part is the response
        parts = cur_round.split(split_word)
        assert len(parts) == 2, "Something went wrong when spliting the prompt."
        inst = parts[0] + split_word
        resp = parts[1] + eos_token

        inst_ids = tokenizer.encode(inst, add_special_tokens=False)
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)

        round_input_ids = inst_ids + resp_ids
        round_label_ids = [label_pad_token_id] * len(inst_ids) + resp_ids
        round_attention_mask = [1] * len(round_input_ids)

        input_ids.extend(round_input_ids)
        label_ids.extend(round_label_ids)
        attention_mask.extend(round_attention_mask)
    
    return dict(
        input_ids=input_ids,
        labels=label_ids,
        attention_mask=attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: transformers.PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.cached_data_dict = {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, List[int]]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        inst = self.data[i]["instruction"]
        resp = self.data[i]["response"]
        messages = [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": resp},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = prompt2inputs(prompt, self.tokenizer, SEPARATOR)

        self.cached_data_dict[i] = inputs
        return inputs


def create_dataset(
    tokenizer: transformers.PreTrainedTokenizerBase,
    data_path: str,
    max_samples: int=None,
):
    logger.info("Loading data...")
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    
    if max_samples is not None:
        max_samples = min(len(data), max_samples)
    data = data[:max_samples]

    dataset = SupervisedDataset(data, tokenizer)

    return dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility before initializing model.
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        use_fast=model_args.use_fast_tokenizer,
        padding_side = "right",
    )
    if "Qwen2Tokenizer" in tokenizer.__class__.__name__:
        tokenizer.bos_token = "<|im_start|>"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set customized chat template for base model
    tokenizer.chat_template = CHAT_TEMPLATE

    logger.info(f"Tokenizer Class: {tokenizer.__class__.__name__}")
    logger.info(f"PAD Token/Id: {tokenizer.pad_token}/{tokenizer.pad_token_id}")
    logger.info(f"BOS Token/Id: {tokenizer.bos_token}/{tokenizer.bos_token_id}")
    logger.info(f"EOS Token/Id: {tokenizer.eos_token}/{tokenizer.eos_token_id}")

    # Load the config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024
    max_pos_embeddings = min(max_pos_embeddings, tokenizer.model_max_length)
    if data_args.max_length is None:
        data_args.max_length = max_pos_embeddings
    
    logger.info(f"Data parameters: {data_args}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")
    
    # Load the model
    bnb_config = None
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    if model_args.use_4bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and model_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if training_args.local_rank == 0 and major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
    
    if model_args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=model_args.use_8bit_quantization)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
    )
    
    model.config.use_cache = not training_args.gradient_checkpointing
    if model_args.use_4bit_quantization or model_args.use_8bit_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if model_args.use_peft_lora:
        target_modules = model_args.lora_target_modules.split(",") if model_args.lora_target_modules != "all-linear" else model_args.lora_target_modules
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        if training_args.local_rank == 0:
            print("=" * 80)
            model.print_trainable_parameters()
            print("=" * 80)
    else:
        all_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info("=" * 80)
        logger.info(f"trainable params: {all_params:,} || all params: {all_params:,} || trainable%: 100")
        logger.info("=" * 80)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Load data
    train_dataset = create_dataset(tokenizer, data_args.data_path, data_args.max_train_samples)

    # Set the data collator
    if data_args.pad_mode == "pad":
        data_collator = DataCollatorWithDynamicPad(
            tokenizer=tokenizer, padding=True, max_length=data_args.max_length,
        )
    elif data_args.pad_mode == "pack":
        data_collator = DataCollatorWithDynamicPack(
            tokenizer=tokenizer, padding=True, max_length=data_args.max_length,
        )
    else:
        raise ValueError(f"Supported padding modes are `pad` and `pack`, but you provided {data_args.pad_mode}")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    # Record GPU usage during training
    trainer.add_callback(MemoryUsageCallback(training_args.local_rank, training_args.device))

    logger.info("=" * 80)
    if model_args.use_peft_lora:
        logger.info(f"PEFT trainable parameters: {trainer.get_num_trainable_parameters():,}")
    else:
        logger.info(f"FULL trainable parameters: {trainer.get_num_trainable_parameters():,}")
    logger.info("=" * 80)

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
     # Save the final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
