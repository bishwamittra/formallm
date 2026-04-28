import os
import pickle

import torch
import numpy as np
import pandas as pd
import evaluate
import psutil
from tqdm import tqdm
from copy import deepcopy
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from nltk.metrics.distance import edit_distance

from utils_encoding import custom_tokenize_string


# ---------------------------------------------------------------------------
# Text-generation callback (generates token-by-token during training)
# ---------------------------------------------------------------------------

class GenereteTextCallback(TrainerCallback):

    def __init__(self,
                 tokenizer,
                 dataset,
                 max_new_tokens,
                 compute_msp,
                 local_prefix_length_list=[5, 10, 20],
                 skip_tokens=20,
                 generation_interval=1,
                 selective_samples=True,
                 global_prefix_config='random_tokens'):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.compute_msp = compute_msp
        self.local_prefix_length_list = local_prefix_length_list
        self.skip_tokens = skip_tokens
        self.generation_interval = generation_interval
        self.selective_samples = selective_samples
        self.global_prefix_config = global_prefix_config
        assert self.global_prefix_config in ['random_token', 'same_language', 'no_global_prefix']

    def remove_eos(self, token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        length = len(token_ids_raw)
        for i, (t, a) in enumerate(zip(token_ids_raw, attentions_raw)):
            if t == self.tokenizer.eos_token_id and i < length - 1:
                continue
            token_ids.append(t)
            attentions.append(a)
        return token_ids, attentions

    def on_step_end(self, args, state, control, **kwargs):
        if args.local_rank != 0:
            return

        if state.epoch % 10 != 0 and state.epoch != 1:
            return

        EPS = 1e-12

        print("Epoch:", state.epoch)

        model = kwargs['model']

        for eval_dataset in self.dataset:
            ground_truth_token_ids_all = []
            prompt_token_ids_all = []
            example_ids = []
            generated_token_ids_all = []
            length_token_ids_all = []

            random_index_list = None
            if self.compute_msp:
                if eval_dataset != "train_sequences":
                    continue

                msp_prefix_length = []
                original_prompt_token_ids = []
                prompt_ids = []
                random_index = []
                generated_token_negative_log_prob_all = []

                if not self.selective_samples:
                    max_index = 20
                    np.random.seed(0)
                    if self.dataset[eval_dataset].shape[0] > max_index:
                        random_index_list = np.random.choice(
                            self.dataset[eval_dataset].shape[0], size=max_index, replace=False
                        )
                    else:
                        random_index_list = np.arange(self.dataset[eval_dataset].shape[0])

                else:
                    sequence_to_index_map = {}
                    for index, token_id in enumerate(self.dataset[eval_dataset]['input_ids']):
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
            for index in tqdm(range(len(self.dataset[eval_dataset]))):
                token_ids_raw, attention_raw = self.dataset[eval_dataset]['input_ids'].tolist()[index], self.dataset[eval_dataset]['attention_mask'].tolist()[index]
                token_ids, attention = self.remove_eos(token_ids_raw, attention_raw)
                token_ids = np.array(token_ids)
                dataset_token_ids.append(token_ids)

            for index in tqdm(range(len(self.dataset[eval_dataset]))):
                if self.compute_msp and index not in random_index_list:
                    continue

                token_ids_raw, attention_raw = self.dataset[eval_dataset]['input_ids'].tolist()[index], self.dataset[eval_dataset]['attention_mask'].tolist()[index]
                token_ids, attention = self.remove_eos(token_ids_raw, attention_raw)
                token_ids = np.array(token_ids)

                prompt_token_ids = []
                token_length = token_ids.shape[0]
                for i in range(1, token_ids.shape[0] - 1):

                    if i % self.generation_interval != 0 or i + self.max_new_tokens > token_length or i <= self.skip_tokens:
                        continue

                    if self.compute_msp:
                        assert self.max_new_tokens == 1
                        for prefix_length in self.local_prefix_length_list + [i]:
                            if prefix_length > i:
                                continue

                            for rand_idx in range(5):

                                if self.global_prefix_config == 'same_language':
                                    dataset_token_ids_sufficient = []
                                    for token_ids_temp in dataset_token_ids:
                                        if len(token_ids_temp) >= i - prefix_length:
                                            dataset_token_ids_sufficient.append(token_ids_temp)

                                    if len(dataset_token_ids_sufficient) == 0:
                                        continue

                                    random_remote_prefix_full = dataset_token_ids_sufficient[np.random.choice(len(dataset_token_ids_sufficient))]
                                    random_remote_prefix = random_remote_prefix_full[:i - prefix_length].copy()

                                elif self.global_prefix_config == 'random_token':
                                    random_remote_prefix = token_ids[:i - prefix_length].copy()
                                    np.random.seed(rand_idx)
                                    np.random.shuffle(random_remote_prefix)

                                elif self.global_prefix_config == 'no_global_prefix':
                                    if rand_idx > 0:
                                        continue
                                    random_remote_prefix = token_ids[i - prefix_length:i - prefix_length].copy()

                                else:
                                    raise ValueError(self.global_prefix_config)

                                local_token_ids = token_ids[i - prefix_length:i]
                                token_ids_perturbed = np.concatenate([random_remote_prefix, local_token_ids])
                                if self.global_prefix_config != 'no_global_prefix':
                                    assert len(token_ids_perturbed) == i
                                prompt_token_ids.append(list(token_ids_perturbed))

                                custom_input = custom_tokenize_string(token_ids_perturbed, attention, len(token_ids_perturbed))
                                for attribute in custom_input:
                                    custom_input[attribute] = custom_input[attribute].to(args.device)

                                hf_output = model.generate(**custom_input,
                                                            max_new_tokens=self.max_new_tokens,
                                                            do_sample=False,
                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                            top_k=None,
                                                            top_p=None,
                                )

                                predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                                ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + self.max_new_tokens]
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
                            custom_input[attribute] = custom_input[attribute].to(args.device)

                        hf_output = model.generate(**custom_input,
                                                    max_new_tokens=self.max_new_tokens,
                                                    do_sample=False,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    top_k=None,
                                                    top_p=None,
                        )

                        predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                        ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + self.max_new_tokens]
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

            if self.compute_msp:
                result['msp_prefix_length'] = msp_prefix_length
                result['original_prompt_token_ids'] = original_prompt_token_ids
                result['prompt_ids'] = prompt_ids
                result['random_index'] = random_index
                result['target_token_negative_log_prob_list'] = generated_token_negative_log_prob_all

            result = pd.DataFrame(result)
            result['eval_dataset'] = eval_dataset
            result['epoch'] = state.epoch

            if not os.path.exists(f"{args.output_dir}/text_generation_result.csv"):
                result.to_csv(f"{args.output_dir}/text_generation_result.csv", index=False)
            else:
                result.to_csv(f"{args.output_dir}/text_generation_result.csv", mode='a', header=False, index=False)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(grammarCallback, selected_token_ids):
    clf_metrics = evaluate.combine(["accuracy"])

    def compute_metrics_for_grammar(eval_preds):

        processed_logits, labels = eval_preds

        preds = processed_logits[:, :, 0]                      # (batch_size, seq_len)
        predicted_token_prob = processed_logits[:, :, 1]       # (batch_size, seq_len)
        target_token_prob = processed_logits[:, :, 2]          # (batch_size, seq_len)
        selected_token_probs = processed_logits[:, :, 3:]      # (batch_size, seq_len, len(selected_token_ids))

        # Shift position of labels. pred position is already shifted in preprocess_logits_for_metrics
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100

        # accuracy
        preds_flatten = preds.flatten()
        shift_labels_flatten = shift_labels.flatten()
        result = clf_metrics.compute(predictions=preds_flatten[mask.flatten()], references=shift_labels_flatten[mask.flatten()])

        # average predicted token prob per token per sequence
        result["predicted_token_prob"] = np.mean(predicted_token_prob[mask])

        # average correct token prob per token per sequence
        result["target_token_prob"] = np.mean(target_token_prob[mask])

        # average total probability mass per sequence per token
        total_prob_mass = np.sum(selected_token_probs, axis=-1)
        result["total_prob_mass"] = np.mean(total_prob_mass[mask])

        store_result_dict = {}
        store_result_dict["label_id"] = shift_labels_flatten
        store_result_dict["pred_id"] = preds_flatten
        store_result_dict["mask"] = mask.flatten()
        store_result_dict["predicted_token_prob"] = predicted_token_prob.flatten()
        store_result_dict["target_token_prob"] = target_token_prob.flatten()
        EPS = 1e-12
        store_result_dict['target_token_negative_log_prob'] = -np.log(target_token_prob.flatten() + EPS)
        store_result_dict["total_prob_mass"] = total_prob_mass.flatten()
        for i, token_id in enumerate(selected_token_ids):
            store_result_dict[f'token_prob_{token_id}'] = selected_token_probs[..., i].flatten()
        grammarCallback.store_result_dict = store_result_dict

        return result

    return compute_metrics_for_grammar


# ---------------------------------------------------------------------------
# Edit-distance helpers
# ---------------------------------------------------------------------------

def process_edit_distance(df):
    # token sequence to sample id
    token_sequence_to_sample_id_map = {}
    for _, row in df.iterrows():
        if row['token_sequence'] not in token_sequence_to_sample_id_map:
            token_sequence_to_sample_id_map[row['token_sequence']] = [row['sample_id']]
        else:
            token_sequence_to_sample_id_map[row['token_sequence']].append(row['sample_id'])

    # pair distance
    token_sequences = df['token_sequence'].unique()
    pair_to_edit_distance = {}
    max_distance = 0
    for token_seq_a in token_sequences:
        for token_seq_b in token_sequences:
            if (token_seq_a, token_seq_b) in pair_to_edit_distance:
                continue
            if (token_seq_b, token_seq_a) in pair_to_edit_distance:
                continue
            distance = edit_distance(token_seq_a, token_seq_b)
            pair_to_edit_distance[(token_seq_a, token_seq_b)] = distance
            pair_to_edit_distance[(token_seq_b, token_seq_a)] = distance
            max_distance = max(max_distance, distance)

    sample_id_to_distance = {}
    for token_seq_a, token_seq_b in pair_to_edit_distance:
        distance = pair_to_edit_distance[(token_seq_a, token_seq_b)]

        for sample_id in token_sequence_to_sample_id_map[token_seq_a]:
            if sample_id not in sample_id_to_distance:
                sample_id_to_distance[sample_id] = {}
                for d in range(max_distance + 1):
                    sample_id_to_distance[sample_id][d] = []
            sample_id_to_distance[sample_id][distance].extend(token_sequence_to_sample_id_map[token_seq_b])

        for sample_id in token_sequence_to_sample_id_map[token_seq_b]:
            if sample_id not in sample_id_to_distance:
                sample_id_to_distance[sample_id] = {}
                for d in range(max_distance + 1):
                    sample_id_to_distance[sample_id][d] = []
            sample_id_to_distance[sample_id][distance].extend(token_sequence_to_sample_id_map[token_seq_a])

    # unique
    for sample_id in sample_id_to_distance:
        total = 0
        for distance in sample_id_to_distance[sample_id]:
            sample_id_to_distance[sample_id][distance] = list(set(sample_id_to_distance[sample_id][distance]))
            total += len(sample_id_to_distance[sample_id][distance])
        assert total == df['sample_id'].nunique()

    return sample_id_to_distance


def drop_low_loss_train_sequences(df, col, percentile=25):
    x = df[col].to_numpy()
    keep_mask = x > np.percentile(x, percentile)
    return df[~keep_mask]


# ---------------------------------------------------------------------------
# Logit preprocessing
# ---------------------------------------------------------------------------

def preprocess_logits(tokenizer, selected_token_ids):
    softmax = torch.nn.Softmax(dim=-1)

    def preprocess_logits_for_metrics(logits, labels):
        """
        Return a tensor (max prob token ids, predicted token prob, correct token prob, selected token probs)
        """
        # shift position of logits
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # logits shape: (batch_size, seq_len, vocab_size)
        pred_ids = torch.argmax(logits, dim=-1)            # (batch_size, seq_len)
        softmax_prob = softmax(logits)                     # (batch_size, seq_len, vocab_size)

        # softmax prob of predicted tokens
        pred_token_probs = softmax_prob.gather(-1, pred_ids.unsqueeze(-1))  # (batch_size, seq_len, 1)

        # softmax prob of correct tokens
        labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
        target_token_probs = softmax_prob.gather(-1, labels.unsqueeze(-1))  # (batch_size, seq_len, 1)

        # softmax prob of selected tokens
        selected_token_probs = softmax_prob[:, :, torch.tensor(selected_token_ids).to(logits.device)]  # (batch_size, seq_len, S)

        # concatenate: pred_ids followed by selected_token_probs
        pred_ids = pred_ids.unsqueeze(-1)
        return_ids = torch.cat([pred_ids,
                                pred_token_probs,
                                target_token_probs,
                                selected_token_probs], dim=-1)

        return return_ids

    return preprocess_logits_for_metrics


# ---------------------------------------------------------------------------
# Grammar evaluation callback
# ---------------------------------------------------------------------------

class GrammarCallback(TrainerCallback):

    def __init__(self, base_config, trainer, tokenizer, dataset, incontext_common_prefix_len, train_test_distance, effective_samples_per_batch, starting_epoch=0):
        self.base_config = base_config
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset.copy()
        self.incontext_input = base_config['incontext_input']
        self.incontext_common_prefix_len = incontext_common_prefix_len
        self.store_result_dict = None
        self.train_test_distance = train_test_distance
        self.intermediate_result = pd.DataFrame()
        self.intermediate_result_earlier_epoch = None
        self.distance_result_train = None
        self.removed_sample_ids_history = {}
        self.starting_epoch = starting_epoch
        self.effective_samples_per_batch = effective_samples_per_batch

        self.optimal_contextual_threshold = {}
        if self.train_test_distance is not None:
            for sample_id in self.train_test_distance:
                self.optimal_contextual_threshold[sample_id] = np.inf
        else:
            for sample_id in range(self.dataset['train_sequences'].num_rows):
                self.optimal_contextual_threshold[sample_id] = 0.2
        self.ignore_sample_ids = None

        if self.base_config['adaptive_training']:
            self.prev_global_step = 0
            self.starting_index = self.effective_samples_per_batch
            self.update_eval_dataset()

        self.process = psutil.Process(os.getpid())

    def get_nearest_training_strings(self, sample_id, max_distance):
        sample_ids_nearest = []
        for d in range(max_distance + 1):
            sample_ids_nearest.extend(self.distance_result_train[sample_id][d])
        return sample_ids_nearest

    def on_evaluate(self, args, state, control, **kwargs):
        # action performed after compute_metrics
        torch.cuda.empty_cache()

        eval_dataset = None
        for key in state.log_history[-1].keys():
            if key.startswith("eval") and key.endswith("loss"):
                eval_dataset = key[5:-5]
                break

        assert eval_dataset is not None, f"{eval_dataset} not found"

        result_dict = self.store_result_dict
        result = pd.DataFrame(result_dict)
        result['epoch'] = state.epoch if state.epoch is not None else 0
        result['epoch'] += self.starting_epoch
        result['global_step'] = state.global_step
        result['eval_dataset'] = eval_dataset
        result['pred_id'] = result['pred_id'].astype(int)
        result['label_id'] = result['label_id'].astype(int)
        result['index_token_ids'] = result.index

        assert eval_dataset in self.trainer.eval_dataset, f"{eval_dataset} not found"
        loader = DataLoader(self.trainer.eval_dataset[eval_dataset],
                            batch_size=args.per_device_eval_batch_size,
                            shuffle=False,
                            collate_fn=self.trainer.data_collator
        )
        length_token_ids = []
        for batch in loader:
            batch_token_ids = batch['input_ids'].cpu().numpy()
            for token_ids in batch_token_ids:
                num_pad_tokens = 0
                for token_id in token_ids:
                    if token_id == self.tokenizer.pad_token_id or token_id == self.tokenizer.bos_token_id:
                        num_pad_tokens += 1
                    else:
                        break
                for i in range(len(token_ids) - 1):
                    length_token_ids.append(max(0, i + 1 - num_pad_tokens))
        result['length_input_tokens'] = length_token_ids

        result['correct'] = result['pred_id'] == result['label_id']

        if self.base_config['memorization_algo'] not in ["no_intervention", "deduplication"]:
            # storing results of training sequences
            self.intermediate_result = pd.concat([self.intermediate_result, result[result['eval_dataset'].isin(
                ['train_sequences', 'test_sequences', 'train_sequences_g1', 'train_sequences_g2', 'test_sequences_g1', 'test_sequences_g2']
            )]]).copy()

        # store once, for local rank 0
        if args.local_rank != 0:
            return

        # store the result
        if not os.path.exists(f"{args.output_dir}/grammar_eval_result.csv"):
            result.to_csv(f"{args.output_dir}/grammar_eval_result.csv", index=False)
        else:
            result.to_csv(f"{args.output_dir}/grammar_eval_result.csv", mode='a', header=False, index=False)

        result_average = result[~result['label_id'].isin([-100])].groupby(['eval_dataset', 'epoch']).aggregate({
            'target_token_negative_log_prob': 'mean',
            'correct': 'mean',
            'total_prob_mass': 'mean'
        }).reset_index()
        if not os.path.exists(f"{args.output_dir}/grammar_eval_result_average.csv"):
            result_average.to_csv(f"{args.output_dir}/grammar_eval_result_average.csv", index=False)
        else:
            result_average.to_csv(f"{args.output_dir}/grammar_eval_result_average.csv", mode='a', header=False, index=False)

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is not None and "loss" in logs:
            logs['seen_samples'] = self.effective_samples_per_batch
            if state.is_local_process_zero:
                if not os.path.exists(f"{args.output_dir}/train_log.csv"):
                    pd.DataFrame([logs]).to_csv(f"{args.output_dir}/train_log.csv", index=False)
                else:
                    pd.DataFrame([logs]).to_csv(f"{args.output_dir}/train_log.csv", mode='a', header=False, index=False)

        self.mat(args, state, control, **kwargs)
        if self.base_config['adaptive_training']:
            if state.global_step > self.prev_global_step:
                self.update_eval_dataset()
                self.prev_global_step = state.global_step

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            max_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            cpu_mem = self.process.memory_info().rss / 1024**2
            if state.is_local_process_zero:
                print(f"CPU mem: {cpu_mem:.2f} MB, GPU mem: {gpu_mem:.2f} GB, max GPU mem: {max_gpu_mem:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

        return

    def update_eval_dataset(self):
        assert "train_sequences" in self.trainer.eval_dataset
        assert "train_sequences" in self.dataset

        min_index = self.starting_index
        max_index = self.starting_index + 2 * self.effective_samples_per_batch
        max_index = min(max_index, self.dataset['train_sequences'].num_rows)

        if min_index >= self.dataset['train_sequences'].num_rows:
            del self.trainer.eval_dataset['train_sequences']
        else:
            self.trainer.eval_dataset['train_sequences'] = self.dataset['train_sequences'].select(range(min_index, max_index))

    def mat(self, args, state, control, **kwargs):
        if self.intermediate_result.shape[0] == 0 or self.incontext_input:
            return

        if self.base_config['memorization_algo'] in ["no_intervention", "deduplication"]:
            return

        if self.intermediate_result['eval_dataset'].nunique() < 2:
            return

        assert self.intermediate_result['epoch'].nunique() == 1
        assert self.intermediate_result['eval_dataset'].nunique() >= 2
        assert "train_sequences" in self.intermediate_result['eval_dataset'].unique()
        assert "test_sequences" in self.intermediate_result['eval_dataset'].unique()

        if args.local_rank == 0:
            print("\n\nEpoch:", self.intermediate_result['epoch'].unique()[0])

        # assign sample id and retrieve token sequence
        self.intermediate_result = self.intermediate_result[~self.intermediate_result['label_id'].isin([-100, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id])]
        list_intermediate_result = []
        for _, df_item in self.intermediate_result.groupby(['eval_dataset']):
            df_item['sample_id'] = (df_item['length_input_tokens'] == 0).cumsum() - 1
            # retrieve token sequence
            list_df_item = []
            for _, df_sample in df_item.groupby('sample_id'):
                df_sample = df_sample.sort_values('length_input_tokens')
                token_sequence = tuple(df_sample['label_id'].values)
                df_sample['token_sequence'] = [token_sequence] * len(df_sample)
                list_df_item.append(df_sample)
            df_item = pd.concat(list_df_item)
            list_intermediate_result.append(df_item)
        self.intermediate_result = pd.concat(list_intermediate_result)
        self.intermediate_result = self.intermediate_result.groupby(['epoch', 'eval_dataset', 'sample_id', 'token_sequence']).aggregate({
            'target_token_negative_log_prob': 'mean',
        }).reset_index()

        # preprocess training sequences by finding edit distance with all other training sequences
        if self.base_config['memorization_algo'].endswith("_edit_distance"):
            self.distance_result_train = process_edit_distance(self.intermediate_result[
                self.intermediate_result['eval_dataset'] == 'train_sequences'
            ])

        # if value is true, then ignore
        self.ignore_sample_ids = {i: False for i in self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences']['sample_id'].unique()}

        if self.base_config['multilingual']:
            # keep track of individual train sequences of different languages
            dict_token_sequence_to_sample_id_map = {}
            dict_ignore_sample_ids = {}
            assert "train_sequences_g1" in self.intermediate_result['eval_dataset'].unique()
            assert "train_sequences_g2" in self.intermediate_result['eval_dataset'].unique()
            for eval_dataset in ["train_sequences_g1", "train_sequences_g2"]:
                if eval_dataset not in dict_token_sequence_to_sample_id_map:
                    dict_token_sequence_to_sample_id_map[eval_dataset] = {}
                if eval_dataset not in dict_ignore_sample_ids:
                    dict_ignore_sample_ids[eval_dataset] = {
                        i: False for i in self.intermediate_result[self.intermediate_result['eval_dataset'] == eval_dataset]['sample_id'].unique()
                    }
                for _, row in self.intermediate_result[self.intermediate_result['eval_dataset'] == eval_dataset].iterrows():
                    if row['token_sequence'] not in dict_token_sequence_to_sample_id_map[eval_dataset]:
                        dict_token_sequence_to_sample_id_map[eval_dataset][row['token_sequence']] = []
                    dict_token_sequence_to_sample_id_map[eval_dataset][row['token_sequence']].append(row['sample_id'])

        if args.local_rank == 0:
            intermediate_result_train = self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences'].copy()
            intermediate_result_test = self.intermediate_result[self.intermediate_result['eval_dataset'] == 'test_sequences'].copy()
            print("Var to mean ratio train:", intermediate_result_train['target_token_negative_log_prob'].std(ddof=0) / intermediate_result_train['target_token_negative_log_prob'].mean())
            print("Var to mean ratio test :", intermediate_result_test['target_token_negative_log_prob'].std(ddof=0) / intermediate_result_test['target_token_negative_log_prob'].mean())

        if self.base_config['memorization_algo'] == "sanity_check":
            for i in self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences']['sample_id'].unique():
                if i == 4:
                    self.ignore_sample_ids[i] = False
                else:
                    self.ignore_sample_ids[i] = True

        elif self.base_config['memorization_algo'] == "tail_distribution":
            token_sequence_to_sample_id_map = {}
            for _, row in self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences'].iterrows():
                if row['token_sequence'] not in token_sequence_to_sample_id_map:
                    token_sequence_to_sample_id_map[row['token_sequence']] = []
                token_sequence_to_sample_id_map[row['token_sequence']].append(row['sample_id'])

            for token_sequence in token_sequence_to_sample_id_map:
                mapped_ids = token_sequence_to_sample_id_map[token_sequence]
                if len(mapped_ids) > 1:
                    for i in mapped_ids:
                        self.ignore_sample_ids[i] = True

        elif self.base_config['memorization_algo'] in ["training_loss_keep_bottom_ref", "training_loss_keep_top_ref", "training_loss_keep_middle_ref"]:
            assert self.base_config['reference_model_results'] is not None
            # check consistency
            with open(f"{self.base_config['reference_model_results']}/args.pkl", 'rb') as f:
                reference_model_args = pickle.load(f)
                assert reference_model_args['memorization_algo'] == "no_intervention"
                for key in ['grammar_name', 'model_name', 'num_samples', 'data_seed', 'run_seed', 'considered_training_samples']:
                    if key not in reference_model_args:
                        raise ValueError(f"self.base_config['{key}'] is not in reference_model_args")
                    if self.base_config[key] != reference_model_args[key]:
                        raise ValueError(f"self.base_config['{key}'] != reference_model_args['{key}']")

            intermediate_result_train = pd.read_csv(
                f"{self.base_config['reference_model_results']}/grammar_eval_result_string_average_optimal_checkpoint.csv"
            )
            intermediate_result_train = intermediate_result_train[intermediate_result_train['eval_dataset'] == 'train_sequences'].copy()
            assert intermediate_result_train.shape[0] == self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences'].shape[0]

            token_sequence_to_sample_id_map = {}
            for _, row in intermediate_result_train.iterrows():
                if row['token_sequence'] not in token_sequence_to_sample_id_map:
                    token_sequence_to_sample_id_map[row['token_sequence']] = []
                token_sequence_to_sample_id_map[row['token_sequence']].append(row['sample_id'])

            intermediate_result_train = intermediate_result_train.drop_duplicates(subset=['token_sequence'])

            if args.local_rank == 0:
                print("Mean loss before", intermediate_result_train['target_token_negative_log_prob'].mean())

            if self.base_config['memorization_algo'] == "training_loss_keep_bottom_ref":
                threshold = intermediate_result_train['target_token_negative_log_prob'].quantile(0.33)
                intermediate_result_train = intermediate_result_train[intermediate_result_train['target_token_negative_log_prob'] < threshold]
            elif self.base_config['memorization_algo'] == "training_loss_keep_top_ref":
                threshold = intermediate_result_train['target_token_negative_log_prob'].quantile(0.67)
                intermediate_result_train = intermediate_result_train[intermediate_result_train['target_token_negative_log_prob'] > threshold]
            elif self.base_config['memorization_algo'] == "training_loss_keep_middle_ref":
                threshold_low = intermediate_result_train['target_token_negative_log_prob'].quantile(0.33)
                threshold_high = intermediate_result_train['target_token_negative_log_prob'].quantile(0.67)
                intermediate_result_train = intermediate_result_train[
                    (intermediate_result_train['target_token_negative_log_prob'] >= threshold_low) &
                    (intermediate_result_train['target_token_negative_log_prob'] <= threshold_high)
                ]

            print("Mean loss after", intermediate_result_train['target_token_negative_log_prob'].mean())

            ignore_token_sequences = intermediate_result_train['token_sequence'].unique()
            for token_sequence in ignore_token_sequences:
                mapped_ids = token_sequence_to_sample_id_map[token_sequence]
                for i in mapped_ids:
                    self.ignore_sample_ids[i] = True

        elif self.base_config['memorization_algo'] in ["training_loss_keep_bottom", "training_loss_keep_top", "training_loss_keep_middle"]:
            intermediate_result_train = self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences'].copy()
            token_sequence_to_sample_id_map = {}
            for _, row in intermediate_result_train.iterrows():
                if row['token_sequence'] not in token_sequence_to_sample_id_map:
                    token_sequence_to_sample_id_map[row['token_sequence']] = []
                token_sequence_to_sample_id_map[row['token_sequence']].append(row['sample_id'])

            intermediate_result_train = intermediate_result_train.drop_duplicates(subset=['token_sequence'])

            if args.local_rank == 0:
                print("Mean loss before", intermediate_result_train['target_token_negative_log_prob'].mean())

            if self.base_config['memorization_algo'] == "training_loss_keep_bottom":
                threshold = intermediate_result_train['target_token_negative_log_prob'].quantile(0.33)
                intermediate_result_train = intermediate_result_train[intermediate_result_train['target_token_negative_log_prob'] < threshold]
            elif self.base_config['memorization_algo'] == "training_loss_keep_top":
                threshold = intermediate_result_train['target_token_negative_log_prob'].quantile(0.67)
                intermediate_result_train = intermediate_result_train[intermediate_result_train['target_token_negative_log_prob'] > threshold]
            elif self.base_config['memorization_algo'] == "training_loss_keep_middle":
                threshold_low = intermediate_result_train['target_token_negative_log_prob'].quantile(0.33)
                threshold_high = intermediate_result_train['target_token_negative_log_prob'].quantile(0.67)
                intermediate_result_train = intermediate_result_train[
                    (intermediate_result_train['target_token_negative_log_prob'] >= threshold_low) &
                    (intermediate_result_train['target_token_negative_log_prob'] <= threshold_high)
                ]

            print("Mean loss after", intermediate_result_train['target_token_negative_log_prob'].mean())

            ignore_token_sequences = intermediate_result_train['token_sequence'].unique()
            for token_sequence in ignore_token_sequences:
                mapped_ids = token_sequence_to_sample_id_map[token_sequence]
                for i in mapped_ids:
                    self.ignore_sample_ids[i] = True

        elif self.base_config['memorization_algo'] == 'training_test_equal_variance':
            if self.intermediate_result_earlier_epoch is not None:
                intermediate_result_train = self.intermediate_result[self.intermediate_result['eval_dataset'] == 'train_sequences'].copy()
                token_sequence_to_sample_id_map = {}
                for _, row in intermediate_result_train.iterrows():
                    if row['token_sequence'] not in token_sequence_to_sample_id_map:
                        token_sequence_to_sample_id_map[row['token_sequence']] = []
                    token_sequence_to_sample_id_map[row['token_sequence']].append(row['sample_id'])

                test_sequences_loss_change = pd.merge(
                    self.intermediate_result_earlier_epoch[self.intermediate_result_earlier_epoch['eval_dataset'] == 'test_sequences'],
                    self.intermediate_result[self.intermediate_result['eval_dataset'] == 'test_sequences'],
                    how='left',
                    on=['sample_id', 'token_sequence'],
                    suffixes=('', '_new')
                )
                test_sequences_loss_change = test_sequences_loss_change.drop_duplicates(subset=['token_sequence'])
                intermediate_result_train = intermediate_result_train.drop_duplicates(subset=['token_sequence'])

                num_affected_test_sequences = test_sequences_loss_change[test_sequences_loss_change['target_token_negative_log_prob_new'] > test_sequences_loss_change['target_token_negative_log_prob']].shape[0]

                if args.local_rank == 0:
                    print("Fraction of test sequences affected:", num_affected_test_sequences / test_sequences_loss_change.shape[0])

                intermediate_result_train = drop_low_loss_train_sequences(intermediate_result_train,
                                                                           'target_token_negative_log_prob',
                                                                           percentile=num_affected_test_sequences / test_sequences_loss_change.shape[0] * 100)
                ignore_token_sequences = intermediate_result_train['token_sequence'].unique()
                for token_sequence in ignore_token_sequences:
                    mapped_ids = token_sequence_to_sample_id_map[token_sequence]
                    for i in mapped_ids:
                        self.ignore_sample_ids[i] = True

                    if self.base_config['multilingual']:
                        for eval_dataset in ["train_sequences_g1", "train_sequences_g2"]:
                            if token_sequence in dict_token_sequence_to_sample_id_map[eval_dataset]:
                                mapped_ids = dict_token_sequence_to_sample_id_map[eval_dataset][token_sequence]
                                for i in mapped_ids:
                                    dict_ignore_sample_ids[eval_dataset][i] = True

        elif "remove" in self.base_config['memorization_algo']:
            in_learning_phase = {}
            for sample_id in self.optimal_contextual_threshold:
                median_threshold = self.intermediate_result[
                    (self.intermediate_result['eval_dataset'] == 'test_sequences') &
                    (self.intermediate_result['sample_id'].isin(self.train_test_distance[sample_id]))
                ]['target_token_negative_log_prob'].median()

                if 'remove_before_memorized' in self.base_config['memorization_algo'] and self.intermediate_result_earlier_epoch is not None:
                    loss_decrease = self.intermediate_result_earlier_epoch[
                        (self.intermediate_result_earlier_epoch['eval_dataset'] == 'train_sequences') &
                        (self.intermediate_result_earlier_epoch['sample_id'] == sample_id)
                    ]['target_token_negative_log_prob'].item() - self.intermediate_result[
                        (self.intermediate_result['eval_dataset'] == 'train_sequences') &
                        (self.intermediate_result['sample_id'] == sample_id)
                    ]['target_token_negative_log_prob'].item()
                    loss_decrease = max(0, loss_decrease)
                    median_threshold += loss_decrease

                if self.optimal_contextual_threshold[sample_id] >= median_threshold:
                    self.optimal_contextual_threshold[sample_id] = median_threshold

            # identify memorized strings
            for _, row in self.intermediate_result[(self.intermediate_result['eval_dataset'] == 'train_sequences')].iterrows():
                if row['target_token_negative_log_prob'] <= self.optimal_contextual_threshold[row['sample_id']]:
                    if row['sample_id'] not in in_learning_phase:
                        if self.base_config['memorization_algo'].endswith("_edit_distance"):
                            for sample_id_near in self.get_nearest_training_strings(row['sample_id'], max_distance=3):
                                self.ignore_sample_ids[sample_id_near] = True
                        else:
                            self.ignore_sample_ids[row['sample_id']] = True

            if "never_put_back" in self.base_config['memorization_algo']:
                for sample_id in self.ignore_sample_ids:
                    if self.ignore_sample_ids[sample_id]:
                        self.removed_sample_ids_history[sample_id] = True
                for sample_id in self.removed_sample_ids_history:
                    self.ignore_sample_ids[sample_id] = True

        elif self.base_config['memorization_algo'] == "manual":
            pass

        else:
            raise ValueError(self.base_config['memorization_algo'])

        if args.local_rank == 0:
            print("Ignoring ids", [i for i in self.ignore_sample_ids if self.ignore_sample_ids[i]])

        total_ignore_sample_ids = sum(self.ignore_sample_ids.values())
        assert len(self.ignore_sample_ids) == self.trainer.eval_dataset['train_sequences'].num_rows

        # update indices
        if total_ignore_sample_ids != self.trainer.eval_dataset['train_sequences'].num_rows:
            if self.base_config['adaptive_training']:

                min_index = self.starting_index
                max_index = self.starting_index + 2 * self.effective_samples_per_batch
                max_index = min(max_index, self.dataset['train_sequences'].num_rows)

                indices_already_considered = [i for i in self.trainer.train_dataset.indices if i < min_index]
                indices_currently_considered = [i + min_index for i in self.ignore_sample_ids.keys() if not self.ignore_sample_ids[i]][:self.effective_samples_per_batch]
                if len(indices_currently_considered) < self.effective_samples_per_batch:
                    indices_currently_considered += list(range(max_index + 1, self.dataset['train_sequences'].num_rows))[:self.effective_samples_per_batch - len(indices_currently_considered)]

                self.starting_index = max(indices_currently_considered) + 1
                indices_will_be_considered = list(range(self.starting_index, self.dataset['train_sequences'].num_rows))

                if args.local_rank == 0:
                    print("Inference: min_index", min_index, "max_index", max_index)
                    print("currently", indices_currently_considered, len(indices_currently_considered))

                self.trainer.train_dataset.indices = indices_already_considered + indices_currently_considered + indices_will_be_considered
                # terminating condition
                if self.starting_index >= self.dataset['train_sequences'].num_rows or len(indices_currently_considered) < self.effective_samples_per_batch:
                    print("Requesting termination", self.starting_index)
                    control.should_training_stop = True

            else:
                self.trainer.train_dataset.indices = [i for i in self.ignore_sample_ids.keys() if not self.ignore_sample_ids[i]]

            if args.local_rank == 0:
                print(f"Pruned {total_ignore_sample_ids} memorization samples.")
                print("\n" * 3)
        else:
            if args.local_rank == 0:
                print(f"All memorization samples pruned.")
                print("\n" * 3)
            control.should_training_stop = True

        # storing pruning results
        self.intermediate_result_earlier_epoch = self.intermediate_result.copy()
        self.intermediate_result = pd.DataFrame()
        if args.local_rank == 0:
            pruning_result = []
            for i in self.ignore_sample_ids:
                pruning_result.append({
                    'eval_dataset': 'train_sequences',
                    'epoch': self.intermediate_result_earlier_epoch['epoch'].iloc[0],
                    'sample_id': i,
                    'is_pruned': self.ignore_sample_ids[i],
                })

            if self.base_config['multilingual']:
                for eval_dataset in ['train_sequences_g1', 'train_sequences_g2']:
                    for i in dict_ignore_sample_ids[eval_dataset]:
                        pruning_result.append({
                            'eval_dataset': eval_dataset,
                            'epoch': self.intermediate_result_earlier_epoch['epoch'].iloc[0],
                            'sample_id': i,
                            'is_pruned': dict_ignore_sample_ids[eval_dataset][i],
                        })

            df_pruning_result = pd.DataFrame(pruning_result)
            if not os.path.exists(f"{args.output_dir}/memorization_pruning.csv"):
                df_pruning_result.to_csv(f"{args.output_dir}/memorization_pruning.csv", index=False)
            else:
                df_pruning_result.to_csv(f"{args.output_dir}/memorization_pruning.csv", mode='a', header=False, index=False)
        return