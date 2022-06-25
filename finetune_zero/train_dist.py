# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import gc
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import fairscale  # pip install fairscale, fairscale-0.3.2
from fairscale.nn.wrap import auto_wrap
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.nn.data_parallel import FullyShardedDataParallel

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.data.processors.squad import SquadV1Processor

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


model = 'mix_LM' # 'mix_LM', 'transformer'
task = 'ICT'  # 'ICT', 'SOP'

parser = argparse.ArgumentParser()

if model == 'mix_LM':
    from models.mix_LM import Config
    parser.add_argument("--config_file", default='models/mix_LM_config.json', type=str, help="")
    if task == 'ICT':
        from models.mix_LM import BertMultiTask_ICT as BertMultiTask
    else:
        from models.mix_LM import BertMultiTask_SOP as BertMultiTask
else:
    from models.transformer import Config
    parser.add_argument("--config_file", default='models/transformer_config.json', type=str, help="")
    if task == 'ICT':
        from models.transformer import BertMultiTask_ICT as BertMultiTask
    else:
        from models.transformer import BertMultiTask_SOP as BertMultiTask

parser.add_argument("--checkpoint", default='pretrain_ckpt/hong_018_50.bin', type=str,
                        help="불러올 사전 학습된 모델의 경로 및 파일명 지정")
parser.add_argument("--output_dir", default='checkpoint', type=str,
                    help="학습 완료 후 모델이 저장될 경로 지정")

## File related parameters
parser.add_argument("--train_file", default='data/KorQuAD_v1.0_dev.json', type=str,
                    help="학습 데이터셋의 경로 및 파일명 지정")

## hyperparameter
parser.add_argument("--max_seq_length", default=384, type=int,
                    help="토크나이저로 토큰화 후 최대 입력 시퀀시 길이 지정")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="문서의 길이가 max_seq_length보다 길때 나누어 입력으로 지정함, 이때 얼마나 건너뛰어 나눌지 지정")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="질문의 최대 길이 지정, 값을 넘는 질문 토큰을 자름")

parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                    help="gpu 당 배치 사이즈 지정.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="백워드 전에 누적할 업데이트 스탭 수를 지정")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="학습률 지정")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="가중치 감쇠율 지정")
parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                    help="Adam 옵티마이저 epsilon 값 지정")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm 값 지정")
parser.add_argument('--seed', type=int, default=42,
                    help="초기화를 위한 랜덤 시드 값 지정")

## Learning procedure parameters
parser.add_argument("--num_train_epochs", default=5.0, type=float,
                    help="총 학습 에포크 수 지정")
parser.add_argument("--warmup_steps", default=50, type=int,
                    help="웜업 스탭 수 지정")

## AUTOMATIC MIXED PRECISION
parser.add_argument('--fp16', action='store_true',
                    help="pytorch 내부 Automatic Mixed Precision 사용 여부 지정")
parser.add_argument('--auto_wrap', action='store_true',
                    help="auto_wrap 사용 여부")
parser.add_argument('--zero3', action='store_true',
                    help="Zero 3 사용 여부, 사용하지 않으면 Zero 2로 작동함, Zero 2가 더 빠름")

args = parser.parse_args()

args.world_size = max(torch.cuda.device_count(), 1)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)
RPC_PORT = 29501
_DATA_PARALLEL_GROUP = None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def training_step(model: torch.nn.Module, batch: list, args: object) -> torch.Tensor:
    
    model.train()
    model.zero_grad()
    
    batch = tuple(t.to(args.device) for t in batch)
    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
    
    if args.fp16:
        with torch.cuda.amp.autocast():
            loss = model(
                input_ids = input_ids,
                token_type_ids = segment_ids, 
                attention_mask = input_mask,
                start_positions = start_positions,
                end_positions = end_positions
            )
            
    else:
        loss = model(
            input_ids = input_ids,
            token_type_ids = segment_ids, 
            attention_mask = input_mask,
            start_positions = start_positions,
            end_positions = end_positions
        )
    
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
        
    if args.fp16:
        model.scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.detach()


def prepare_dataloader(rank: int, Q: object, args: object) -> (object, int):
    
    loading_file = args.corpus_path + '/' + Q.get()
    max_len = int(args.max_steps_per_epoch * args.train_batch_size)
    train_dataset = LMDataset(corpus_path=loading_file,
                          tokenizer=args.tokenizer,
                          local_rank=rank,
                          max_len=max_len,
                          seq_len=args.max_seq_length,
                          vocab_size=args.vocab_size)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=True
    )

    return train_dataloader, max_len
    
    
def prepare_model(args: object) -> torch.nn.Module:
    
    model = BertMultiTask(args.config, train_mode='mrc')

    if args.checkpoint != None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))

    if args.auto_wrap:
        model = auto_wrap(model)
        
    model = FullyShardedDataParallel(
        model,
        mixed_precision=args.fp16,
        flatten_parameters=True,
        reshard_after_forward=args.zero3).to(args.device)
    
    if args.fp16:
        model.scaler = ShardedGradScaler()

    return model


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

    
    
def train(
    rank: int,
    world_size: int,
    train_dataset: object,
    args: object):
    global _DATA_PARALLEL_GROUP
    
    # process group init
    torch.cuda.set_device(rank)
    args.device = torch.device("cuda", rank)
    args.n_gpu = 1
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:{}".format(RPC_PORT), rank=rank, world_size=world_size)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # model init
    model = prepare_model(args)
    
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    optimizer = AdamW(params=optimizer_grouped_parameters,
                  lr=args.learning_rate,
                  eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                 num_warmup_steps=t_total * 0.1,
                                 num_training_steps=t_total)

    global_step = 0
    iter = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=rank != 0)
    
    for _ in train_iterator:
        
        epoch_iterator = tqdm(
            train_dataloader, 
            desc="Train(XX Epoch) Step(XXXX/XXXX) (loss=XX.XXXXX)", 
            disable=rank != 0)
        
        tr_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            
            if (step + 1) % args.gradient_accumulation_steps != 0:
                with model.no_sync():
                    loss = training_step(model, batch, args)
            else:
                loss = training_step(model, batch, args)
                
            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.fp16:
                    model.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    model.scaler.step(optimizer)
                    scale = model.scaler.get_scale()
                    model.scaler.update()
                    skip_lr_sched = (scale != model.scaler.get_scale())
                    if not skip_lr_sched:
                        scheduler.step()  # Update learning rate schedule
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                
                global_step += 1

                epoch_iterator.set_description(
                    "Train(%d Epoch) Step(%d / %d) (loss=%5.5f)" % (_, global_step, t_total, loss))
                
        model_state_dict = model.state_dict()
        
        if rank == 0:
            logger.info("** ** * Saving file * ** **")
            model_checkpoint = 'pe_{0}_{1}_{2}_{3}.bin'.format(args.learning_rate,
                                                              args.train_batch_size,
                                                              int(args.num_train_epochs),
                                                              _)
            
            logger.info(model_checkpoint)
            output_model_file = os.path.join(args.output_dir, model_checkpoint)
            torch.save(model_state_dict, output_model_file)

    logger.info("Training End!!!")

def load_and_cache_examples(args, tokenizer):

    # Load data features from cache or dataset file
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}'.format('train', str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        processor = SquadV1Processor()
        input_a = input_file.split('/')
        examples = processor.get_train_examples(''.join(input_a[:-1]), filename=input_a[-1])
        features, _ = transformers.squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=4,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)

    return dataset
    
    
def main():
    global args

    # Set seed
    set_seed(args)
    
    # Set Configuration
    args.config = Config(args.config_file)
    args.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    args.vocab_size = args.tokenizer.vocab_size
    args.config.max_seq_length = args.max_seq_length
    
    train_dataset = load_and_cache_examples(args, args.tokenizer)
    
    # Training
    mp.spawn(
        train,
        args=(args.world_size, train_dataset, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
