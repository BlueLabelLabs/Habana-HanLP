# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:52
import os
import random
import time
from typing import List, Union, Dict, Tuple
import numpy as np
import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown, nvmlDeviceGetCount
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from hanlp.utils.io_util import get_resource, replace_ext, TimingFileIterator
from hanlp.utils.log_util import logger, flash
from hanlp_common.constant import HANLP_VERBOSE
from hanlp_common.io import load_pickle, save_pickle

# Add imports for Habana support
try:
    import habana_frameworks.torch.core as htcore

    HABANA_AVAILABLE = True
except ImportError:
    HABANA_AVAILABLE = False


def hpus_available() -> Dict[int, float]:
    """Check for available HPUs

    Returns:
        Dict mapping HPU IDs to availability ratio (1.0 for all HPUs as memory info not available via API)
    """
    if not HABANA_AVAILABLE:
        return dict()

    try:
        # Get visible HPU devices
        visible_devices = os.environ.get('HABANA_VISIBLE_DEVICES', None)
        num_devices = htcore.get_device_count()

        if num_devices == 0:
            return dict()

        if visible_devices is None:
            visible_devices = list(range(num_devices))
        else:
            visible_devices = {int(x.strip()) for x in visible_devices.split(',')}

        # For Habana devices, we don't have memory info API like NVIDIA
        # So we just return 1.0 as the ratio for all available devices
        hpus = {i: 1.0 for i, real_id in enumerate(visible_devices)}
        return dict(sorted(hpus.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.debug(f'Failed to get HPU info due to {e}')
        return dict()


def gpus_available() -> Dict[int, float]:
    if not torch.cuda.is_available():
        return dict()
    try:
        nvmlInit()
        gpus = {}
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices is None:
            visible_devices = list(range(nvmlDeviceGetCount()))
        else:
            visible_devices = {int(x.strip()) for x in visible_devices.split(',')}
        for i, real_id in enumerate(visible_devices):
            h = nvmlDeviceGetHandleByIndex(real_id)
            info = nvmlDeviceGetMemoryInfo(h)
            total = info.total
            free = info.free
            ratio = free / total
            gpus[i] = ratio
        nvmlShutdown()
        return dict(sorted(gpus.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.debug(f'Failed to get gpu info due to {e}')
        return dict((i, 1.0) for i in range(torch.cuda.device_count()))


def get_available_devices() -> Dict[str, Dict[int, float]]:
    """Get all available devices (CUDA GPUs and HPUs)

    Returns:
        Dict mapping device type to device availability info
    """
    devices = {}
    gpus = gpus_available()
    if gpus:
        devices['cuda'] = gpus

    hpus = hpus_available()
    if hpus:
        devices['hpu'] = hpus

    return devices


def cuda_devices(query=None) -> List[int]:
    """Decide which GPUs to use

    Args:
      query: GPU selection criteria (None, int, float, or list)

    Returns:
      List of GPU IDs to use
    """
    if isinstance(query, list):
        if len(query) == 0:
            return [-1]
        return query
    if query is None:
        query = gpus_available()
        if not query:
            return []
        size, idx = max((v, k) for k, v in query.items())
        # When multiple GPUs have the same size, randomly pick one to avoid conflicting
        gpus_with_same_size = [k for k, v in query.items() if v == size]
        query = random.choice(gpus_with_same_size)
    if isinstance(query, float):
        gpus = gpus_available()
        if not gpus:
            return []
        query = [k for k, v in gpus.items() if v > query]
    elif isinstance(query, int):
        query = [query]
    return query


def hpu_devices(query=None) -> List[int]:
    """Decide which HPUs to use

    Args:
      query: HPU selection criteria (None, int, float, or list)

    Returns:
      List of HPU IDs to use
    """
    if isinstance(query, list):
        if len(query) == 0:
            return [-1]
        return query
    if query is None:
        query = hpus_available()
        if not query:
            return []
        size, idx = max((v, k) for k, v in query.items())
        # When multiple HPUs have the same size, randomly pick one to avoid conflicting
        hpus_with_same_size = [k for k, v in query.items() if v == size]
        query = random.choice(hpus_with_same_size)
    if isinstance(query, float):
        hpus = hpus_available()
        if not hpus:
            return []
        query = [k for k, v in hpus.items() if v > query]
    elif isinstance(query, int):
        query = [query]
    return query


def get_device_str(device_type='auto', device_id=0):
    """Get device string for torch

    Args:
        device_type: 'auto', 'cpu', 'cuda', or 'hpu'
        device_id: device ID

    Returns:
        device string for torch
    """
    if device_type == 'auto':
        devices = get_available_devices()
        if 'hpu' in devices and devices['hpu']:
            return 'hpu'
        elif 'cuda' in devices and devices['cuda']:
            return f'cuda:{device_id}'
        else:
            return 'cpu'
    elif device_type == 'hpu':
        if not HABANA_AVAILABLE:
            logger.warning("HPU requested but Habana libraries not available. Falling back to CPU.")
            return 'cpu'
        return 'hpu'
    elif device_type == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        return f'cuda:{device_id}'
    else:
        return 'cpu'


def set_seed(seed=233, dont_care_speed=False):
    """Copied from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L76
    Args:
      seed:  (Default value = 233)
      dont_care_speed: True may have a negative single-run performance impact, but ensures deterministic
    Returns:
        """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.cuda.manual_seed_all(seed)
    if HABANA_AVAILABLE:
        # Set Habana specific seeds if available
        htcore.manual_seed(seed)
    if dont_care_speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_to_device(tensor, device):
    """Move tensor to specific device (cpu, cuda, or hpu)

    Args:
        tensor: PyTorch tensor
        device: device string or torch.device

    Returns:
        tensor on specified device
    """
    if isinstance(device, str) and device == 'hpu' and HABANA_AVAILABLE:
        return tensor.to('hpu')
    else:
        return tensor.to(device)


def clip_grad_norm(model: nn.Module, grad_norm, transformer: nn.Module = None, transformer_grad_norm=None):
    if transformer_grad_norm is None:
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_norm)
    else:
        is_transformer = []
        non_transformer = []
        transformer = set(transformer.parameters())
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p in transformer:
                is_transformer.append(p)
            else:
                non_transformer.append(p)
        nn.utils.clip_grad_norm_(non_transformer, grad_norm)
        nn.utils.clip_grad_norm_(is_transformer, transformer_grad_norm)


def main():
    start = time.time()

    # Print available CUDA GPUs
    print("Available GPUs:", gpus_available())

    # # Print available Habana HPUs
    # if HABANA_AVAILABLE:
    #     print("Available HPUs:", hpus_available())
    # else:
    #     print("Habana framework not available")
    #
    # # Get all available devices
    # print("All available devices:", get_available_devices())
    #
    # # Print auto-selected device
    # print("Auto-selected device:", get_device_str())
    #
    # print(f"Time elapsed: {time.time() - start:.4f}s")


if __name__ == '__main__':
    main()


def clip_grad_norm(model: nn.Module, grad_norm, transformer: nn.Module = None, transformer_grad_norm=None):
    if transformer_grad_norm is None:
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_norm)
    else:
        is_transformer = []
        non_transformer = []
        transformer = set(transformer.parameters())
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p in transformer:
                is_transformer.append(p)
            else:
                non_transformer.append(p)
        nn.utils.clip_grad_norm_(non_transformer, grad_norm)
        nn.utils.clip_grad_norm_(is_transformer, transformer_grad_norm)


def load_word2vec(path, delimiter=' ', cache=True) -> Tuple[Dict[str, np.ndarray], int]:
    realpath = get_resource(path)
    binpath = replace_ext(realpath, '.pkl')
    if cache:
        try:
            flash('Loading word2vec from cache [blink][yellow]...[/yellow][/blink]')
            word2vec, dim = load_pickle(binpath)
            flash('')
            return word2vec, dim
        except IOError:
            pass

    dim = None
    word2vec = dict()
    f = TimingFileIterator(realpath)
    for idx, line in enumerate(f):
        f.log('Loading word2vec from text file [blink][yellow]...[/yellow][/blink]')
        line = line.rstrip().split(delimiter)
        if len(line) > 2:
            if dim is None:
                dim = len(line)
            else:
                if len(line) != dim:
                    logger.warning('{}#{} length mismatches with {}'.format(path, idx + 1, dim))
                    continue
            word, vec = line[0], line[1:]
            word2vec[word] = np.array(vec, dtype=np.float32)
    dim -= 1
    if cache:
        flash('Caching word2vec [blink][yellow]...[/yellow][/blink]')
        save_pickle((word2vec, dim), binpath)
        flash('')
    return word2vec, dim


def load_word2vec_as_vocab_tensor(path, delimiter=' ', cache=True) -> Tuple[Dict[str, int], torch.Tensor]:
    realpath = get_resource(path)
    vocab_path = replace_ext(realpath, '.vocab')
    matrix_path = replace_ext(realpath, '.pt')
    if cache:
        try:
            if HANLP_VERBOSE:
                flash('Loading vocab and matrix from cache [blink][yellow]...[/yellow][/blink]')
            vocab = load_pickle(vocab_path)
            matrix = torch.load(matrix_path, map_location='cpu')
            if HANLP_VERBOSE:
                flash('')
            return vocab, matrix
        except IOError:
            pass

    word2vec, dim = load_word2vec(path, delimiter, cache)
    vocab = dict((k, i) for i, k in enumerate(word2vec.keys()))
    matrix = torch.Tensor(np.stack(list(word2vec.values())))
    if cache:
        flash('Caching vocab and matrix [blink][yellow]...[/yellow][/blink]')
        save_pickle(vocab, vocab_path)
        torch.save(matrix, matrix_path)
        flash('')
    return vocab, matrix


def save_word2vec(word2vec: dict, filepath, delimiter=' '):
    with open(filepath, 'w', encoding='utf-8') as out:
        for w, v in word2vec.items():
            out.write(f'{w}{delimiter}')
            out.write(f'{delimiter.join(str(x) for x in v)}\n')


def lengths_to_mask(seq_len, max_len=None):
    r"""
    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param torch.LongTensor seq_len: (B,)
    :param int max_len: max sequence length。
    :return:  torch.Tensor  (B, max_len)
    """
    assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
    batch_size = seq_len.size(0)
    max_len = int(max_len) if max_len else seq_len.max().long()
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))

    return mask


def activation_from_name(name: str):
    return getattr(torch.nn, name)


def filter_state_dict_safely(model_state: dict, load_state: dict):
    safe_state = dict()
    for k, v in load_state.items():
        model_v = model_state.get(k, None)
        if model_v is not None and model_v.shape == v.shape:
            safe_state[k] = v
    return safe_state
