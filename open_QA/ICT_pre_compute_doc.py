from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm
import time

from IR.mix_LM import Config as IRConfig, BertMultiTask_ICT as BertMultiTask

import transformers
from utils.ICT_pretrain_utils import WikiDatasetInference
from functools import reduce

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def InformationRetrieval():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='pre_compute_vector', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='checkpoint/pe_5e-05_126_5_4.bin',
                        type=str,
                        help="checkpoint")
    parser.add_argument("--model_config", default='IR/mix_LM_config.json', type=str)
    parser.add_argument("--corpus_file", default='wiki_corpus/ko_wiki_sep_by_paragraph_preproc.dat', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    
    
    parser.add_argument("--predict_batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu,  args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # Prepare model
    config = IRConfig.from_json_file(args.model_config)
    config.max_seq_length = args.max_seq_length
    model = BertMultiTask(config, train_mode='ict')
    model.load_state_dict(torch.load(args.checkpoint))
    if args.fp16:
        model.half()
    model.to(device)
    eval_examples = WikiDatasetInference(corpus_path=args.corpus_file, tokenizer=tokenizer, query_len=args.max_query_length, doc_len=args.max_seq_length)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.predict_batch_size)

    eval_sampler = SequentialSampler(eval_examples)
    eval_dataloader = DataLoader(eval_examples, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_embeddings = []
    all_paragraphs = []
    logger.info("Start evaluating")
    start_time = time.time()
    for (input_ids, input_mask), titles, documents in tqdm(eval_dataloader, desc="Pre-Computing"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        #segment_ids = segment_ids.to(device)
        with torch.no_grad():
            c_encode = model.bert(input_ids = input_ids, attention_mask = input_mask)
            c_encode_pooled = model.decoder(c_encode)
        all_embeddings.append(c_encode_pooled.detach().cpu())
        for title, document in zip(titles, documents):
            res = [title] + [document]
            all_paragraphs.append(res)
        #print(all_paragraphs)
    end_time = time.time()
    embedding_vectors = reduce(lambda x, y: torch.cat([x, y]), all_embeddings)
    logger.info("Pre-computation time: {} ms".format((end_time-start_time)*0.001))
    logger.info("Embedding Vector Size: {}".format(embedding_vectors.size()))
    logger.info("Saving Embedding Vectors and Paragraphs...")
    embedding_save_path = os.path.join(args.output_dir, "pre-compute.vec")
    paragraph_save_path = os.path.join(args.output_dir, "paragraph.pkl")
    torch.save(embedding_vectors, embedding_save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    with open(paragraph_save_path, "wb") as w:
        pickle.dump(all_paragraphs, w, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    InformationRetrieval()
