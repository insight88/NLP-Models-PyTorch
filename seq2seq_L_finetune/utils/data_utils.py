import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class DiagDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=32):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        self.datasets = []
        for i, data in enumerate(tqdm(dataset, desc="Preprocessing")):
            src_tokens = tokenizer.tokenize(data[0])
            src_tokens = [tokenizer.cls_token] + src_tokens[:max_len-2] + ['[EOS]']
            src_ids = [tokenizer.convert_tokens_to_ids(t) for t in src_tokens]
            src_mask = [1] * len(src_ids)

            tgt_tokens = tokenizer.tokenize(data[1])
            tgt_tokens = [tokenizer.cls_token] + tgt_tokens[:max_len-2] + ['[EOS]']
            tgt_ids = [tokenizer.convert_tokens_to_ids(t) for t in tgt_tokens]
            tgt_mask = [1] * len(tgt_ids)

            # Zero Padding
            src_n_pad = max_len - len(src_ids)
            tgt_n_pad = max_len - len(tgt_ids)
            src_ids.extend([0] * src_n_pad)
            src_mask.extend([0] * src_n_pad)
            tgt_ids.extend([0] * tgt_n_pad)
            tgt_mask.extend([0] * tgt_n_pad)

            # Target Mask
            tgt_sub_mask = np.tril(np.ones((max_len, max_len))).astype(bool)
            tgt_mask_seq = tgt_mask & tgt_sub_mask
            tgt_mask_seq = tgt_mask_seq[:-1, :-1]

            tensors = (torch.tensor(src_ids, dtype=torch.long),
                       torch.tensor(tgt_ids, dtype=torch.long),
                       torch.tensor(src_mask, dtype=torch.long),
                       torch.tensor(tgt_mask_seq, dtype=torch.long))
            self.datasets.append(tensors)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return self.datasets[item]


class DiagDataset_Test(Dataset):
    def __init__(self, path, tokenizer, max_len=32):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        self.datasets = []
        self.labels = []
        for i, data in enumerate(tqdm(dataset, desc="Preprocessing")):
            src_tokens = [tokenizer.cls_token] + tokenizer.tokenize(data[0]) + ['[EOS]']
            src_ids = [tokenizer.convert_tokens_to_ids(t) for t in src_tokens]
            assert len(src_ids) < max_len
            src_mask = [1] * len(src_ids)

            # Zero Padding
            src_n_pad = max_len - len(src_ids)
            src_ids.extend([0] * src_n_pad)
            src_mask.extend([0] * src_n_pad)

            tensors = (torch.tensor(src_ids, dtype=torch.long),
                       torch.tensor(src_mask, dtype=torch.long))
            self.datasets.append(tensors)
            self.labels.append(data[1])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return self.datasets[item], self.labels[item]