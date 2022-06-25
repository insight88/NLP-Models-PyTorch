import logging
import torch
import os
import pickle

import numpy as np

from random import randint, random, shuffle

from torch.utils.data import Dataset
from tqdm import tqdm
from utils.masking_utils_seq import _sample_mask

logger = logging.getLogger(__name__)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        if len(tokens_a) + len(tokens_b) <= max_num_tokens:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_instances(documents, max_len, local_rank, short_seq_prob=0.1):
    instances = []
    current_chunk = []
    current_length = 0
    max_len = max_len - 3 # [CLS] [SEP] [EOS]
    for i, doc in enumerate(tqdm(documents, desc="Create Instances", unit=" doc", disable=local_rank not in [-1, 0])):
        seq_len = max_len
        if random() < short_seq_prob:
            seq_len = randint(2, max_len)
        for d in doc:
            current_chunk.append(d)
            current_length += len(d)
            if current_length >= seq_len:
                if current_chunk and len(current_chunk) >= 2:
                    a_end = randint(1, len(current_chunk)-1)

                    tokens_a = []
                    for index in range(a_end):
                        tokens_a.extend(current_chunk[index])

                    tokens_b = []
                    for index in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[index])

                    instance = (tokens_a, tokens_b)
                    instances.append(instance)
                    current_chunk = []
                    current_length = 0

    return instances


class LMDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, local_rank, max_len=0, seq_len=512, vocab_size=32000):
        self.tokenizer = tokenizer
        self.max_len = seq_len
        self.docs = []
        self.mask_prob = 0.3
        self.max_predictions_per_seq = 80
        self.vocab_size = vocab_size
        self.local_rank = local_rank
        # self.epoch = 0
        doc = []
        new_line_check = 0
        num_line = 0
        logger.info("LMDataset init...")
        cached_features_file = corpus_path + '.cache'
        if os.path.exists(cached_features_file):
            logger.info("Loading dataset from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as reader:
                self.docs = pickle.load(reader)
        else:
            with open(corpus_path, 'r', encoding='utf-8')as f:
                for line in tqdm(f, desc="Loading Dataset", unit=" lines", disable=local_rank not in [-1, 0]):
                    line = line.strip()
                    if line == "" or line =="":
                        if num_line <= 1 and new_line_check == 0:
                            doc = []
                            new_line_check +=1
                            num_line = 0
                        elif new_line_check == 0:
                            self.docs.append(doc)
                            doc = []
                            new_line_check += 1
                            num_line = 0
                        else:
                            continue
                    else:
                        tokens = tokenizer.tokenize(line)
                        doc.append(tokens)
                        new_line_check = 0
                        num_line += 1

                if doc:
                    self.docs.append(doc)  # If the last doc didn't end on a newline, make sure it still gets added

                if len(self.docs) <= 1:
                    exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                         "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                         "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                         "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                         "sections or paragraphs.")

                logger.info("Saving dataset into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.docs, writer, protocol=pickle.HIGHEST_PROTOCOL)

        instance = create_instances(self.docs, self.max_len, self.local_rank)
        shuffle(instance)
        if max_len != 0:
            self.instance = instance[:max_len]
        else:
            self.instance = instance
        

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, item):
        instance = self.instance[item]

        tokens_a = instance[0]
        tokens_b = instance[1]

        truncate_seq_pair(tokens_a, tokens_b, self.max_len-3)
        #assert len(tokens_a) >= 1
        #assert len(tokens_b) >= 1
        
        # SOP, sentence-order prediction
        if random() < 0.5:
            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token] + tokens_b + ['[EOS]']
        else:
            tokens = [self.tokenizer.cls_token] + tokens_b + [self.tokenizer.sep_token] + tokens_a + ['[EOS]']
        
        n_prob = min(self.max_predictions_per_seq, max(1, int(round(len(tokens) * self.mask_prob))))
        _tokens, masked_lm_labels = _sample_mask(seg=tokens, tokenizer=self.tokenizer, mask_alpha=4,
                                                mask_beta=1, max_gram=7, goal_num_predict=n_prob)

        src_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in _tokens]
        tgt_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]
        
        src_mask = [1]*len(_tokens)
        tgt_mask = [1]*len(tokens)

        # Zero Padding
        src_n_pad = self.max_len - len(_tokens)
        n_pad = self.max_len - len(tokens)
        src_ids.extend([0]*src_n_pad)
        tgt_ids.extend([0]*n_pad)
        src_mask.extend([0]*src_n_pad)
        tgt_mask.extend([0]*n_pad)
        
        # Target Mask
        tgt_sub_mask = np.tril(np.ones((self.max_len, self.max_len))).astype(bool)
        tgt_mask_seq = tgt_mask & tgt_sub_mask
        tgt_mask_seq = tgt_mask_seq[:-1, :-1]
        
        tensors = (torch.tensor(src_ids, dtype=torch.long),
                       torch.tensor(tgt_ids, dtype=torch.long),
                       torch.tensor(src_mask, dtype=torch.long),
                       torch.tensor(tgt_mask_seq, dtype=torch.long))

        return tensors
