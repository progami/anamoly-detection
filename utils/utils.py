import torch
import random
import numpy as np
import os
from config.search_space import *


def fix_seed(seed=19):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def translate_sequence(sequence):
    i = 0
    dec_seq = []
    while i<len(sequence):
        block = sequence[i: i+number_features_per_block]
        i += number_features_per_block
        dec_block = {
            "conv_n_out_channels,": conv_n_out_channels[block[0]],
            "conv_kernel_size,": conv_kernel_size[block[1]],
            "pool_type,": pool_type[block[2]],
            "pool_kernel_size": pool_kernel_size[block[3]],
            "norm_type,": norm_type[block[4]],
            "act_funcs,": act_funcs[block[5]]
        }
        dec_seq.append(dec_block)
    return dec_seq

