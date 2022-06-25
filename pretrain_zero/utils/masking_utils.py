import numpy as np
import collections
import random

def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, tokenizer, vocab_size, ngram=3, masked_lm_prob=0.15,
                                 max_predictions_per_seq=80):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if (len(cand_indexes) >= 1 and not _is_start_piece(token)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if _is_start_piece(token):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor longer ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                               pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            if output_tokens[index] == tokenizer.cls_token or output_tokens[index] == tokenizer.sep_token:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = tokenizer.mask_token
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = tokenizer.convert_ids_to_tokens(random.randint(0, vocab_size-1))
            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    random.shuffle(ngram_indexes)
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return output_tokens, masked_lm_positions, masked_lm_labels


def _sample_mask(seg, tokenizer, mask_alpha, mask_beta,
                 max_gram=3, goal_num_predict=80):
    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    # 3-gram implementation
    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True) # p(n) = 1/n / sigma(1/k)

    cur_len = 0

    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)

        # `mask_alpha` : number of tokens forming group
        # `mask_beta` : number of tokens to be masked in each groups.
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx

        while beg < seg_len and not _is_start_piece([seg[beg]]):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece([seg[beg]]):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    tokens, mask_labels = [], []
    for i in range(seg_len):
        if mask[i] and (seg[i] != tokenizer.cls_token and seg[i] != tokenizer.sep_token):
            mask_labels.append(tokenizer.convert_tokens_to_ids(seg[i]))
            tokens.append(tokenizer.mask_token)
        else:
            mask_labels.append(-1)
            tokens.append(seg[i])

    return tokens, mask_labels