import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from transformers.cache_utils import Cache

from utils_encoding import custom_tokenize_string


def compute_inference_results(model,
                               tokenizer,
                               dataset,
                               selected_token_ids,
                               incontext_common_prefix_len,
                               store_path,
                               device,
                               batch_size=4,
):

    def remove_eos_and_add_bos(token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        for t, a in zip(token_ids_raw, attentions_raw):
            if t in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]:
                continue
            token_ids.append(t)
            attentions.append(a)

        token_ids.insert(0, tokenizer.bos_token_id)
        attentions.insert(0, 0)
        return token_ids, attentions

    def find_cut_off(input_ids_prompt):
        non_interested_token_count = 0
        for token_id in input_ids_prompt:
            if token_id not in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
                break
            non_interested_token_count += 1
        return non_interested_token_count

    def pad_batch(batch, max_length):
        batch_processed = {
            "input_ids": [],
            "attention_mask": []
        }
        for input_ids, attention_mask in batch:
            num_pad = max_length - len(input_ids)
            batch_processed["input_ids"].append(input_ids + [tokenizer.eos_token_id] * num_pad)
            batch_processed["attention_mask"].append(attention_mask + [0] * num_pad)
        return batch_processed

    def build_prefix_cache_single(prefix_ids, prefix_mask):
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([prefix_mask], dtype=torch.long, device=device)

        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        return out

    def repeat_cache(cache_obj, repeat_times):
        cache_copy = deepcopy(cache_obj)
        if hasattr(cache_copy, "batch_repeat_interleave"):
            cache_copy.batch_repeat_interleave(repeat_times)
            return cache_copy

        if isinstance(cache_copy, tuple):
            return tuple(
                (
                    key.expand(repeat_times, -1, -1, -1).clone(),
                    value.expand(repeat_times, -1, -1, -1).clone()
                )
                for key, value in cache_copy
            )

        return cache_copy

    def guided_generation(batch_target, max_length):
        effective_batch_size = len(batch_target["input_ids"])

        selected_token_probs = [[] for _ in range(effective_batch_size)]
        pred_token_ids = [[] for _ in range(effective_batch_size)]
        target_token_probs = [[] for _ in range(effective_batch_size)]
        target_token_negative_log_probs = [[] for _ in range(effective_batch_size)]
        predicted_token_probs = [[] for _ in range(effective_batch_size)]

        target_ids_tensor = torch.tensor(
            batch_target["input_ids"], dtype=torch.long, device=device
        )  # [B, T]

        selected_ids_tensor = torch.tensor(
            selected_token_ids, dtype=torch.long, device=device
        )

        prefix_cache = repeat_cache(initialization["prefix_past_key_values"], effective_batch_size)
        logits = initialization["prefix_logits"].expand(effective_batch_size, -1).clone()

        past_key_values = prefix_cache

        with torch.inference_mode():
            for idx in range(max_length):
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)

                current_target_ids = target_ids_tensor[:, idx]  # [B]

                pred_ids = torch.argmax(logits, dim=-1)  # [B]
                pred_probs = probs.gather(1, pred_ids.unsqueeze(1)).squeeze(1)  # [B]

                tgt_probs = probs.gather(1, current_target_ids.unsqueeze(1)).squeeze(1)  # [B]
                tgt_nll = -log_probs.gather(1, current_target_ids.unsqueeze(1)).squeeze(1)  # [B]

                sel_probs = probs.index_select(dim=1, index=selected_ids_tensor)  # [B, S]

                for b in range(effective_batch_size):
                    target_token_negative_log_probs[b].append(tgt_nll[b].item())
                    target_token_probs[b].append(tgt_probs[b].item())
                    pred_token_ids[b].append(pred_ids[b].item())
                    predicted_token_probs[b].append(pred_probs[b].item())

                    selected_token_prob = {}
                    for k, selected_token_id in enumerate(selected_token_ids):
                        selected_token_prob[selected_token_id] = sel_probs[b, k].item()
                    selected_token_probs[b].append(selected_token_prob)

                next_input_ids = current_target_ids.unsqueeze(1)  # [B, 1]

                with torch.inference_mode():
                    step_out = model(
                        input_ids=next_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )

                past_key_values = step_out.past_key_values
                logits = step_out.logits[:, -1, :]

        return (
            target_token_negative_log_probs,
            target_token_probs,
            selected_token_probs,
            pred_token_ids,
            predicted_token_probs,
        )

    representative_eval_dataset = "incontext_common_prefix"
    if representative_eval_dataset in dataset:
        assert dataset[representative_eval_dataset].num_rows == 1
        input_ids_incontext = dataset[representative_eval_dataset][0]["input_ids"].tolist()
        attention_mask_incontext = dataset[representative_eval_dataset][0]["attention_mask"].tolist()
        input_ids_incontext, attention_mask_incontext = remove_eos_and_add_bos(
            input_ids_incontext, attention_mask_incontext
        )
    else:
        input_ids_incontext = [tokenizer.bos_token_id]
        attention_mask_incontext = [0]

    initialization = {}

    prefix_out = build_prefix_cache_single(input_ids_incontext, attention_mask_incontext)
    initialization["prefix_past_key_values"] = prefix_out.past_key_values
    initialization["prefix_logits"] = prefix_out.logits[:, -1, :]
    initialization["input_ids_incontext"] = input_ids_incontext
    initialization["attention_mask_incontext"] = attention_mask_incontext

    for eval_dataset in dataset:
        list_grammar_eval_result = []
        if eval_dataset == representative_eval_dataset:
            continue

        for i in tqdm(range(0, dataset[eval_dataset].num_rows, batch_size)):
            batch = dataset[eval_dataset][i:i + batch_size]

            batch_processed = []
            max_length = 0
            length_orig = []
            for j in range(len(batch["input_ids"])):
                input_ids = batch["input_ids"][j].tolist()
                attention_mask = batch["attention_mask"][j].tolist()
                cut_off = find_cut_off(input_ids)
                input_ids_target = input_ids[cut_off:]
                attention_mask_target = attention_mask[cut_off:]
                max_length = max(max_length, len(input_ids_target))
                length_orig.append(len(input_ids_target))
                batch_processed.append((input_ids_target, attention_mask_target))

            batch_processed_with_pad = pad_batch(batch_processed, max_length)

            (
                target_token_negative_log_probs,
                target_token_probs,
                output_selected_token_probs,
                pred_token_ids,
                predicted_token_probs,
            ) = guided_generation(batch_processed_with_pad, max_length)

            for idx_batch in range(len(length_orig)):
                j = 0
                for label_id, \
                    pred_id, \
                    predicted_token_prob, \
                    target_token_prob, \
                    target_token_negative_log_prob, \
                    selected_token_probs in zip(
                        batch_processed_with_pad["input_ids"][idx_batch],
                        pred_token_ids[idx_batch],
                        predicted_token_probs[idx_batch],
                        target_token_probs[idx_batch],
                        target_token_negative_log_probs[idx_batch],
                        output_selected_token_probs[idx_batch]
                    ):

                    if j >= length_orig[idx_batch]:
                        break

                    result = {}
                    result["label_id"] = label_id
                    result["pred_id"] = pred_id
                    result["predicted_token_prob"] = predicted_token_prob
                    result["target_token_prob"] = target_token_prob
                    result["target_token_negative_log_prob"] = target_token_negative_log_prob

                    sum_selected_token_probs = 0
                    for selected_token_id in selected_token_ids:
                        result[f"token_prob_{selected_token_id}"] = selected_token_probs[selected_token_id]
                        sum_selected_token_probs += selected_token_probs[selected_token_id]

                    result["total_prob_mass"] = sum_selected_token_probs
                    result["mask"] = None
                    result["epoch"] = 0
                    result["global_step"] = 0
                    result["eval_dataset"] = eval_dataset
                    result["index_token_ids"] = i
                    result["length_input_tokens"] = incontext_common_prefix_len + j
                    result["correct"] = result["label_id"] == result["pred_id"]

                    list_grammar_eval_result.append(result)
                    j += 1

        df_grammar_eval_result = pd.DataFrame(list_grammar_eval_result)
        os.makedirs(store_path, exist_ok=True)
        out_csv = f"{store_path}/grammar_eval_result.csv"
        if not os.path.exists(out_csv):
            df_grammar_eval_result.to_csv(out_csv, index=False)
        else:
            df_grammar_eval_result.to_csv(out_csv, mode="a", header=False, index=False)

    return


def text_generation(
    model,
    tokenizer,
    dataset,
    comment,
    device,
    output_dir,
    eval_dataset="train_sequences",
    max_new_tokens=1,
    compute_msp=True,
    local_prefix_length_list=[5, 10, 20],
    skip_tokens=0,
    generation_interval=1,
    selective_samples=True
):

    def remove_eos(token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        length = len(token_ids_raw)
        for i, (t, a) in enumerate(zip(token_ids_raw, attentions_raw)):
            if t == tokenizer.eos_token_id and i < length - 1:
                continue
            token_ids.append(t)
            attentions.append(a)
        return token_ids, attentions

    EPS = 1e-12
    ground_truth_token_ids_all = []
    prompt_token_ids_all = []
    example_ids = []
    generated_token_ids_all = []
    length_token_ids_all = []

    random_index_list = None

    if compute_msp:
        msp_prefix_length = []
        original_prompt_token_ids = []
        prompt_ids = []
        random_index = []
        generated_token_negative_log_prob_all = []
        np.random.seed(0)
        if not selective_samples:
            max_samples_considered = 3
            if dataset[eval_dataset].shape[0] > max_samples_considered:
                random_index_list = np.random.choice(
                    dataset[eval_dataset].shape[0], size=max_samples_considered, replace=False
                )
            else:
                random_index_list = np.arange(dataset[eval_dataset].shape[0])
        else:
            sequence_to_index_map = {}
            for index, token_id in enumerate(dataset[eval_dataset]['input_ids']):
                token_id = tuple(token_id.cpu().numpy())
                if token_id not in sequence_to_index_map:
                    sequence_to_index_map[token_id] = []
                sequence_to_index_map[token_id].append(index)

            sequence_freq = {}
            for sequence in sequence_to_index_map:
                sequence_freq[sequence] = len(sequence_to_index_map[sequence])
            sequence_freq = dict(sorted(sequence_freq.items(), key=lambda x: x[1], reverse=True))

            # take max, median, and min
            max_idx = 0
            min_idx = -1
            median_idx = len(sequence_freq) // 2
            random_index_list = [sequence_to_index_map[list(sequence_freq.keys())[max_idx]][0],
                                  sequence_to_index_map[list(sequence_freq.keys())[median_idx]][0],
                                  sequence_to_index_map[list(sequence_freq.keys())[min_idx]][0]]

        print(f"Random index list: {random_index_list}")

    dataset_token_ids = []
    for index in tqdm(range(len(dataset[eval_dataset]))):
        token_ids_raw, attention_raw = dataset[eval_dataset]['input_ids'].tolist()[index], dataset[eval_dataset]['attention_mask'].tolist()[index]
        token_ids, attention = remove_eos(token_ids_raw, attention_raw)
        token_ids = np.array(token_ids)
        dataset_token_ids.append(token_ids)

    for index in tqdm(range(len(dataset[eval_dataset]))):
        if compute_msp and index not in random_index_list:
            continue

        token_ids_raw, attention_raw = dataset[eval_dataset]['input_ids'].tolist()[index], dataset[eval_dataset]['attention_mask'].tolist()[index]
        token_ids, attention = remove_eos(token_ids_raw, attention_raw)
        token_ids = np.array(token_ids)

        prompt_token_ids = []
        token_length = token_ids.shape[0]
        for i in range(1, token_ids.shape[0] - 1):

            if i % generation_interval != 0 or i + max_new_tokens > token_length or i <= skip_tokens:
                continue

            if compute_msp:
                assert max_new_tokens == 1
                for prefix_length in local_prefix_length_list + [i]:
                    if prefix_length > i:
                        continue

                    for rand_idx in range(5):
                        dataset_token_ids_sufficient = []
                        for token_ids_temp in dataset_token_ids:
                            if len(token_ids_temp) >= i - prefix_length:
                                dataset_token_ids_sufficient.append(token_ids_temp)

                        if len(dataset_token_ids_sufficient) == 0:
                            continue

                        random_remote_prefix_full = dataset_token_ids_sufficient[np.random.choice(len(dataset_token_ids_sufficient))]
                        random_remote_prefix = random_remote_prefix_full[:i - prefix_length].copy()
                        local_token_ids = token_ids[i - prefix_length:i]
                        token_ids_perturbed = np.concatenate([random_remote_prefix, local_token_ids])
                        prompt_token_ids.append(list(token_ids_perturbed))

                        custom_input = custom_tokenize_string(token_ids_perturbed, attention, i)
                        for attribute in custom_input:
                            custom_input[attribute] = custom_input[attribute].to(device)

                        hf_output = model.generate(**custom_input,
                                                    max_new_tokens=max_new_tokens,
                                                    do_sample=False,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    top_k=None,
                                                    top_p=None,
                        )

                        predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                        ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + max_new_tokens]
                        min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                        negative_log_prob = []
                        for new_token_idx in range(min_length):
                            all_token_probs = torch.nn.functional.softmax(hf_output['scores'][new_token_idx][0], dim=0).cpu().numpy()
                            token_prob = all_token_probs[ground_truth_token_ids[new_token_idx]]
                            negative_log_prob.append(-np.log(token_prob + EPS))

                        if min_length == 0:
                            continue
                        predicted_token_ids = predicted_token_ids[:min_length]
                        ground_truth_token_ids = ground_truth_token_ids[:min_length]
                        negative_log_prob = negative_log_prob[:min_length]

                        # store values
                        if i == prefix_length:
                            msp_prefix_length.append("full")
                        else:
                            msp_prefix_length.append(prefix_length)
                        random_index.append(rand_idx)
                        prompt_ids.append(i)
                        length_token_ids_all.append(i)
                        original_prompt_token_ids.append(list(token_ids[:i]))
                        ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                        prompt_token_ids_all.append(prompt_token_ids[-1])
                        example_ids.append(index)
                        generated_token_ids_all.append(list(predicted_token_ids))
                        generated_token_negative_log_prob_all.append(negative_log_prob)

                        if prefix_length == i:
                            break
            else:
                prompt_token_ids.append(list(token_ids[:i]))
                custom_input = custom_tokenize_string(token_ids, attention, i)
                for attribute in custom_input:
                    custom_input[attribute] = custom_input[attribute].to(device)

                hf_output = model.generate(**custom_input,
                                            max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id,
                                            top_k=None,
                                            top_p=None,
                )

                predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + max_new_tokens]
                min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                if min_length == 0:
                    continue
                predicted_token_ids = predicted_token_ids[:min_length]
                ground_truth_token_ids = ground_truth_token_ids[:min_length]

                # store values
                length_token_ids_all.append(i)
                ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                prompt_token_ids_all.append(prompt_token_ids[-1])
                example_ids.append(index)
                generated_token_ids_all.append(list(predicted_token_ids))

    result = {
        "example_ids": example_ids,
        "prompt_token_ids": prompt_token_ids_all,
        "generated_token_ids": generated_token_ids_all,
        "ground_truth_token_ids": ground_truth_token_ids_all,
        "length_input_tokens": length_token_ids_all
    }

    if compute_msp:
        result['msp_prefix_length'] = msp_prefix_length
        result['original_prompt_token_ids'] = original_prompt_token_ids
        result['prompt_ids'] = prompt_ids
        result['random_index'] = random_index
        result['target_token_negative_log_prob_list'] = generated_token_negative_log_prob_all

    result = pd.DataFrame(result)
    result['eval_dataset'] = eval_dataset
    result['comment'] = comment

    if not os.path.exists(f"{output_dir}/text_generation_result.csv"):
        result.to_csv(f"{output_dir}/text_generation_result.csv", index=False)
    else:
        result.to_csv(f"{output_dir}/text_generation_result.csv", mode='a', header=False, index=False)


def min_distant_sequences(train_sequences, test_sequences, distance_dict):
    result = {}
    for i, train_sequence in enumerate(train_sequences):
        train_sequence = tuple(train_sequence)
        distance_list = []
        for test_sequence in test_sequences:
            test_sequence = tuple(test_sequence)
            distance_list.append(abs(distance_dict[test_sequence] - distance_dict[train_sequence]))
        result[i] = []
        min_distance = min(distance_list)
        for j in range(len(test_sequences)):
            if distance_list[j] == min_distance:
                result[i].append(j)
        if len(result[i]) == 0:
            print(distance_list)
            print(min(distance_list))
    return result
