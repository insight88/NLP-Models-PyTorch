from __future__ import absolute_import, division, print_function

import os
import math
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils.data_utils import DiagDataset
from models.seq_seq import Config, Transformer

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--output_dir", default='ckpt', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_file", default='models/seq_seq_config.json', type=str, 
                        help="모델 설정 파일 지정")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/hong_003_50000.bin', type=str,
                        help="불러올 사전 학습된 모델의 경로 및 파일명 지정")

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=2000, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    logger.info('Init Tokenizer...')
    tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    special_tokens_dict = {'additional_special_tokens': ['[EOS]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    logger.info('Data loading...')
    train_dataset = DiagDataset('data/diag_train.pkl', tokenizer=tokenizer)
    val_dataset = DiagDataset('data/diag_val.pkl', tokenizer=tokenizer)

    logger.info('Prepare model...')
    config = Config(args.config_file)
    model = Transformer(config)
    model.load_state_dict(torch.load(args.checkpoint))
    
    model.to(args.device)

    logger.info('{}'.format(config))
    logger.info('Params:{}'.format(count_parameters(model)))

    total_steps = len(train_dataset) / args.batch_size * args.num_train_epochs
    logger.info('Total steps:{}'.format(total_steps))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    

    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    val_sampler = SequentialSampler(val_dataset)
    valid_iterator = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    best_valid_loss = float('inf')
    for epoch in range(args.num_train_epochs):
        # Train!
        epoch_iterator = tqdm(train_iterator, desc="Training (XX Epochs) (total_loss=XX.XXXXX)", disable=args.local_rank not in [-1, 0])
        total_loss = 0.0
        model.zero_grad()
        model.train()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            src_ids, tgt_ids, src_mask, tgt_mask = batch

            loss = model(src_ids, tgt_ids[:,:-1], src_mask, tgt_mask, tgt_ids[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            epoch_iterator.set_description(
                "Training (%d Epochs) (Total loss=%2.5f)"
                % (epoch, loss))
            total_loss += loss.item()

        step_loss = total_loss / step
        logger.info("Train Loss: {} | Train PPL: {}".format(step_loss, math.exp(step_loss)))

        # Eval!
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(valid_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                src_ids, tgt_ids, src_mask, tgt_mask = batch

                loss = model(src_ids, tgt_ids[:, :-1], src_mask, tgt_mask, tgt_ids[:, 1:])
                eval_loss += loss.item()

        e_loss = eval_loss / step
        e_ppl = math.exp(e_loss)
        logger.info("Eval Loss: {} | Eval PPL: {}".format(e_loss, e_ppl))

        if e_ppl < best_valid_loss:
            best_valid_ppl = e_ppl
            path = os.path.join(args.output_dir, 'best_%.5f.pt' % best_valid_ppl)
            torch.save(model.state_dict(), path)
        else:
            logger.info("Early Stop!")
            logger.info("Training End...")


if __name__ == "__main__":
    main()
