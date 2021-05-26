import os
import math
import torch
import pyhocon
import numpy as np
import tempfile
from boltons.iterutils import pairwise, windowed

def prepare_configs(config_name, verbose=True):
    configs = pyhocon.ConfigFactory.parse_file('configs/basic.conf')[config_name]
    configs['saved_path'] = 'trained'
    if not os.path.exists(configs['saved_path']):
        os.makedirs(configs['saved_path'])
    if verbose: print(configs)
    return configs

def flatten(l):
    return [item for sublist in l for item in sublist]

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def bucket_distance(distances, nb_buckets=15):
    """
    Places the given values (designed for distances) into semi-logscale buckets.
    For example if nb_buckets = 15 then:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64-127, 128-255, 256-511, 512-1023, 1024-2047, 2048+].
    """
    logspace_idx = torch.floor(torch.log2(distances.float())).long() + 3
    use_identity = (distances <= 4).long()
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return torch.clamp(combined_idx, 0, nb_buckets-1)

def extract_input_masks_from_mask_windows(mask_windows):
    input_masks = []
    for mask_window in mask_windows:
        subtoken_count = listRightIndex(mask_window, -3) + 1
        input_masks.append([1] * subtoken_count + [0] * (len(mask_window) - subtoken_count))
    input_masks = np.array(input_masks)
    return input_masks

def convert_to_sliding_window(expanded_tokens, sliding_window_size, tokenizer):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param expanded_tokens:
    :param sliding_window_size:
    :return:
    """
    CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])
    SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])
    expanded_masks = [1] * len(expanded_tokens)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = CLS + original_tokens + SEP + PAD * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    return token_windows, mask_windows

def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)
