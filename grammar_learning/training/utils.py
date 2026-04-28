# utils.py — re-export shim for backwards compatibility.
#
# All logic has been moved to focused modules:
#   utils_args.py       — argument parsing, separator_dict
#   utils_data.py       — data loading and preprocessing
#   utils_encoding.py   — tokenization and encoding helpers
#   utils_callbacks.py  — TrainerCallback subclasses, metrics, logit preprocessing
#   utils_generation.py — inference / text generation utilities
#   utils_logging.py    — experiment logging and path management
#   utils_trainer.py    — NoShuffleTrainer
#
# Existing callers continue to work unchanged via this file.

from utils_args import get_args, separator_dict
from utils_logging import set_path, get_logger
from utils_encoding import (
    get_tokenizer,
    tokenize,
    characterwise_encoding,
    encode_dataset,
    custom_tokenize_string,
    custom_tokenize_string_batch,
    get_selected_token_ids,
)
from utils_data import (
    get_data,
    prepare_input_for_incontext,
    process_for_under_trained_tokens,
)
from utils_callbacks import (
    GenereteTextCallback,
    GrammarCallback,
    compute_metrics,
    preprocess_logits,
    drop_low_loss_train_sequences,
    process_edit_distance,
)
from utils_generation import (
    compute_inference_results,
    text_generation,
    min_distant_sequences,
)
from utils_trainer import NoShuffleTrainer

__all__ = [
    # args
    "get_args",
    "separator_dict",
    # logging
    "set_path",
    "get_logger",
    # encoding
    "get_tokenizer",
    "tokenize",
    "characterwise_encoding",
    "encode_dataset",
    "custom_tokenize_string",
    "custom_tokenize_string_batch",
    "get_selected_token_ids",
    # data
    "get_data",
    "prepare_input_for_incontext",
    "process_for_under_trained_tokens",
    # callbacks
    "GenereteTextCallback",
    "GrammarCallback",
    "compute_metrics",
    "preprocess_logits",
    "drop_low_loss_train_sequences",
    "process_edit_distance",
    # generation
    "compute_inference_results",
    "text_generation",
    "min_distant_sequences",
    # trainer
    "NoShuffleTrainer",
]