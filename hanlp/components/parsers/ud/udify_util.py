# This file is modified from udify and allennlp, which are licensed under the MIT license:
# MIT License
#
# Copyright (c) 2019 Dan Kondratyuk and allennlp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from typing import List, Dict, Tuple, Union
import numpy
import torch


def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """Retrieves all treebank data paths in the given directory.
    Adopted from https://github.com/Hyperparticle/udify
    MIT Licence
    Args:
      dataset_dir:
      treebanks:
      dataset_dir: str:
      treebanks: List[str]:  (Default value = None)
    Returns:
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]
        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]
        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None
        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets


def sequence_cross_entropy(log_probs: torch.FloatTensor,
                           targets: torch.LongTensor,
                           weights: torch.FloatTensor,
                           average: str = "batch",
                           label_smoothing: float = None) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")
    device = log_probs.device
    dtype = log_probs.dtype

    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = log_probs.view(-1, log_probs.size(2))
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long().to(device=log_probs.device)

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = log_probs.size(-1)
        smoothing_value = label_smoothing / num_classes

        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)

    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    negative_log_likelihood = negative_log_likelihood * weights.float().to(device=log_probs.device)

    if average == "batch":
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float().to(log_probs.device) + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum().to(log_probs.device) + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float().to(log_probs.device) + 1e-13)
    else:
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float().to(log_probs.device) + 1e-13)
        return per_batch_loss


def sequence_cross_entropy_with_logits(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        weights: Union[torch.FloatTensor, torch.BoolTensor],
        average: str = "batch",
        label_smoothing: float = None,
        gamma: float = None,
        alpha: Union[float, List[float], torch.FloatTensor] = None, ) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    device = logits.device
    dtype = logits.dtype

    weights = weights.to(dtype=dtype, device=device)
    non_batch_dims = tuple(range(1, len(weights.shape)))
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    targets_flat = targets.view(-1, 1).long().to(device=device)

    if gamma:
        probs_flat = log_probs_flat.exp()
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        focal_factor = (1.0 - probs_flat) ** gamma
        weights *= focal_factor.view(*targets.size())

    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_factor = torch.tensor([1.0 - alpha, alpha], dtype=dtype, device=device)
        else:
            alpha_factor = torch.tensor(alpha, dtype=dtype, device=device)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights *= alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -(log_probs_flat * smoothed_targets).sum(-1, keepdim=True)
    else:
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)

    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size()) * weights

    if average == "batch":
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + tiny_value_of_dtype(dtype))
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(dtype)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + tiny_value_of_dtype(dtype))
    else:
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + tiny_value_of_dtype(dtype))
        return per_batch_loss


def tiny_value_of_dtype(dtype: torch.dtype):
    """Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in [torch.float, torch.double]:
        return 1e-13
    elif dtype == torch.half or dtype == torch.bfloat16:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
