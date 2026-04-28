import torch
import transformers
from transformers import AutoTokenizer


def get_tokenizer(args, load_tokenizer=True):
    # if load_tokenizer is false, it means we are interested in only knowing the
    # checkpoint_path, which should be the original model path not the fine-tuned one
    if args.checkpoint_path_overwrite is None or not load_tokenizer:
        checkpoint_path = args.model_name
    else:
        checkpoint_path = args.checkpoint_path_overwrite

    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, checkpoint_path
    else:
        return checkpoint_path


def tokenize(tokenizer, text, max_length):
    if tokenizer.padding_side == "right":
        print("Padding side is right, setting it to left")
        tokenizer.padding_side = "left"
    if max_length is None:
        padding = "longest"
    else:
        padding = "max_length"
    return tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        padding=padding,
        max_length=max_length,
    )


def characterwise_encoding(tokenizer,
                            dataset,
                            max_length,
                            verbose=False,
                            instruction=None):
    sequences = dataset["text"]
    sequence_token_ids = []
    sequence_token_masks = []
    for sequence in sequences:
        sequence_chars = list(sequence)

        encoded_chars = tokenize(
            tokenizer,
            sequence_chars,
            max_length=4
        )

        if instruction is None:
            num_padding = max_length - len(sequence) + 1  # +1 for end of sentence token
            padded_input_ids = torch.cat(
                (
                    torch.tensor([tokenizer.pad_token_id] * num_padding, dtype=torch.long),
                    torch.tensor([tokenizer.bos_token_id] * 1, dtype=torch.long),
                    encoded_chars.input_ids[:, -1:].squeeze(1),
                    torch.tensor([tokenizer.eos_token_id] * 1, dtype=torch.long),
                )
            )
            padded_attention_mask = torch.cat(
                (
                    torch.tensor([0] * num_padding, dtype=torch.long),
                    torch.tensor([0] * 1, dtype=torch.long),
                    encoded_chars.attention_mask[:, -1:].squeeze(1),
                    torch.tensor([1] * 1, dtype=torch.long),
                )
            )
        else:
            instruction_tokens = tokenizer.encode(instruction)
            instruction_attention_mask = [1] * len(instruction_tokens)
            num_padding = max_length - len(sequence) - len(instruction_tokens) + 1  # +1 for end of sentence token
            padded_input_ids = torch.cat(
                (
                    torch.tensor([tokenizer.pad_token_id] * num_padding, dtype=torch.long),
                    torch.tensor([tokenizer.bos_token_id] * 1, dtype=torch.long),
                    torch.tensor(instruction_tokens, dtype=torch.long),
                    encoded_chars.input_ids[:, -1:].squeeze(1),
                    torch.tensor([tokenizer.eos_token_id] * 1, dtype=torch.long),
                )
            )
            padded_attention_mask = torch.cat(
                (
                    torch.tensor([0] * num_padding, dtype=torch.long),
                    torch.tensor([0] * 1, dtype=torch.long),
                    torch.tensor(instruction_attention_mask, dtype=torch.long),
                    encoded_chars.attention_mask[:, -1:].squeeze(1),
                    torch.tensor([1] * 1, dtype=torch.long),
                )
            )

        sequence_token_ids.append(padded_input_ids)
        sequence_token_masks.append(padded_attention_mask)

    if verbose:
        print({
            "input_ids": torch.stack(sequence_token_ids),
            "attention_mask": torch.stack(sequence_token_masks),
        })
        print(torch.stack(sequence_token_ids).shape)

    return {
        "input_ids": torch.stack(sequence_token_ids),
        "attention_mask": torch.stack(sequence_token_masks),
    }


def encode_dataset(tokenizer,
                   dataset,
                   max_sequence_length,
                   verbose=True,
                   instruction_data=None):

    for split_name, split_dataset in dataset.items():
        dataset[split_name] = split_dataset.map(
            lambda batch: characterwise_encoding(
                tokenizer=tokenizer,
                dataset=batch,
                max_length=max_sequence_length,
                verbose=verbose,
                instruction=instruction_data['instruction'] if instruction_data['add_instruction'] and split_name == "incontext_common_prefix" else None
            ),
            batched=True,
            batch_size=128,
        )
    return dataset


def custom_tokenize_string(tokens, attention, position):
    custom_input = transformers.tokenization_utils_base.BatchEncoding()
    custom_input['input_ids'] = []
    custom_input['attention_mask'] = []
    for tvalue, avalue in zip(tokens[:position], attention[:position]):
        custom_input['input_ids'].append(tvalue)
        custom_input['attention_mask'].append(avalue)
    custom_input['input_ids'] = torch.tensor(custom_input['input_ids']).unsqueeze(0)
    custom_input['attention_mask'] = torch.tensor(custom_input['attention_mask']).unsqueeze(0)
    return custom_input


def custom_tokenize_string_batch(tokens, attention, pad_token_id, max_position=None):
    custom_input = transformers.tokenization_utils_base.BatchEncoding()
    custom_input['input_ids'] = []
    custom_input['attention_mask'] = []
    if max_position is None:
        max_position = len(tokens)
    assert max_position >= 1

    for i in range(1, max_position):
        num_padding = max_position - i
        custom_input['input_ids'].append([pad_token_id] * num_padding + list(tokens[:i]))
        custom_input['attention_mask'].append([0] * num_padding + list(attention[:i]))

    custom_input['input_ids'] = torch.tensor(custom_input['input_ids'])
    custom_input['attention_mask'] = torch.tensor(custom_input['attention_mask'])
    return custom_input


def get_selected_token_ids(tokenizer, unique_tokens):
    selected_token_ids = tokenize(
        tokenizer,
        unique_tokens,
        max_length=4
    ).input_ids[:, -1:].squeeze(1).tolist()
    return selected_token_ids