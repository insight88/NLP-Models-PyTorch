# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange

from IR.mix_LM import Config, BertMultiTask_ICT as BertMultiTask
#from IR.transformer import Config, BertMultiTask_ICT as BertMultiTask

from IR.qca_utils import QCASampler

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def prepare_optimizer_parameters(args, model):

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = QCASampler(train_dataset, args.train_batch_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    optimizer = AdamW(params=optimizer_grouped_parameters,
                  lr=args.learning_rate,
                  eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                 num_warmup_steps=t_total * 0.1,
                                 num_training_steps=t_total)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    model.zero_grad()
    model.train()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XXXX/XXXX) (loss=XX.XXXXX)", disable=args.local_rank not in [-1, 0])
        
        temp_s, temp_c = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)  # multi-gpu does scattering it-self
            no, q_input_ids, q_input_mask, c_input_ids, c_input_mask, use_answer = batch
            
            loss = model(
                input_ids = q_input_ids,
                attention_mask = q_input_mask, 
                c_input_ids = c_input_ids, 
                c_attention_mask = c_input_mask, 
                use_answer = use_answer
            )  

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Train(%d Epoch) Step(%d / %d) (loss=%5.5f)" % (_, global_step, t_total, loss.tolist()))

        logger.info("** ** * Saving file * ** **")
        model_checkpoint = 'pe_{0}_{1}_{2}_{3}.bin'.format(args.learning_rate,
                                                              args.train_batch_size,
                                                              int(args.num_train_epochs),
                                                              _)
        logger.info(model_checkpoint)
        output_model_file = os.path.join(args.output_dir, model_checkpoint)
        if args.n_gpu > 1:
            torch.save(model.module.state_dict(), output_model_file)
        else:
            torch.save(model.state_dict(), output_model_file)

    logger.info("Training End!!!")

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def features_parse(features, dataset, tokenizer, max_query_length, max_seq_length):
    
    doc_no = 0
    para_d = ""
    ft = []
    for d, f in zip(dataset, features):
        x = (d[0] * d[2]).tolist()
        c_ids = remove_values_from_list(x, 0)
        c_input_ids = [tokenizer.cls_token_id] + c_ids
        c_input_mask = [1] * len(c_input_ids)

        x = (d[0] * torch.logical_xor(d[1], d[2]).long()).tolist()
        q_ids = remove_values_from_list(x, 0)
        q_input_ids = q_ids
        q_input_mask = [1] * len(q_input_ids)

        document = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_input_ids[:max_seq_length - max_query_length]))
        if para_d != document:  # 문서가 바뀌었는지 체크하여 문서 번호 체크
            para_d = document
            doc_no += 1

        while len(c_input_ids) < max_seq_length:
            c_input_ids.append(0)
            c_input_mask.append(0)

        while len(q_input_ids) < max_query_length:
            q_input_ids.append(0)
            q_input_mask.append(0)

        if len(q_input_ids) > max_query_length or len(c_input_ids) > max_seq_length:
            continue

        if f.is_impossible:
            continue
        ft.append([doc_no, q_input_ids, q_input_mask, c_input_ids, c_input_mask, not f.is_impossible])
 
    return ft
    
def load_and_cache_examples(args, tokenizer):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}'.format('train',
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        processor = SquadV2Processor()
        
        input_a = input_file.split('/')
        examples = processor.get_train_examples(''.join(input_a[:-1]), filename=input_a[-1])
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length - 2,
            is_training=True,
            return_dataset="pt",
            threads=4,
        )
        del examples
        
        features = features_parse(features, dataset, tokenizer, args.max_query_length, args.max_seq_length)
        del dataset
        
        if args.add_train_file_0:
            input_a = args.add_train_file_0.split('/')
            examples = processor.get_train_examples(''.join(input_a[:-1]), filename=input_a[-1])
            features_0, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length - 2,
                is_training=True,
                return_dataset="pt",
                threads=4,
            )
            del examples
            
            features_0 = features_parse(features_0, dataset, tokenizer, args.max_query_length, args.max_seq_length)
            del dataset
            
            features_1 = features[:]
            del features
            
            features = []
            for f in features_0:
                features.append(f)
            del features_0
            
            for f in features_1:
                features.append(f)
            del features_1

            
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()

    #빈문서 세트 만들어 넣어 주기
    n_doc_id = 0
    n_que_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
    n_que_mask = [1] * len(n_que_ids)
    while len(n_que_ids) < args.max_query_length:
        n_que_ids.append(0)
        n_que_mask.append(0)
    
    n_ans_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
    n_ans_mask = [1] * len(n_ans_ids)
    while len(n_ans_ids) < args.max_seq_length:
        n_ans_ids.append(0)
        n_ans_mask.append(0)
    n_use_answer = False
    
    features.insert(0, [n_doc_id, n_que_ids, n_que_mask, n_ans_ids, n_ans_mask, n_use_answer])  # 빈문서 추가
    
    doc_id = torch.tensor([f[0] for f in features], dtype=torch.long)
    q_input_ids = torch.tensor([f[1] for f in features], dtype=torch.long)
    q_input_mask = torch.tensor([f[2] for f in features], dtype=torch.long)
    c_input_ids = torch.tensor([f[3] for f in features], dtype=torch.long)
    c_input_mask = torch.tensor([f[4] for f in features], dtype=torch.long)
    use_answer = torch.tensor([f[5] for f in features], dtype=torch.bool)
    dataset = TensorDataset(doc_id, q_input_ids, q_input_mask, c_input_ids, c_input_mask, use_answer)
    
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default='pretrain_ckpt/hong_384_50000.bin',
                        type=str,
                        help="checkpoint")
    parser.add_argument("--output_dir", default='checkpoint', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default='data/KorQuAD_v1.0_train.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--add_train_file_0", default='data/ko_nia_normal_squad_all.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    
    parser.add_argument("--config_name", default="IR/mix_LM_config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    parser.add_argument("--per_gpu_train_batch_size", default=42, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=20, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.config = Config.from_json_file(args.config_name)
    
    args.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    args.vocab_size = args.tokenizer.vocab_size
    
    args.config.vocab_size = args.vocab_size
    args.config.max_seq_length = args.max_seq_length
    
    
    model = BertMultiTask(args.config, train_mode='ict')
    model.load_state_dict(torch.load(args.checkpoint))
    torch.nn.init.normal_(model.decoder.linear.weight, std=args.config.initializer_range)
    
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, args.tokenizer)
    train(args, train_dataset, model)

if __name__ == "__main__":
    main()
