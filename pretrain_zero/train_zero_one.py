# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import gc
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler

import fairscale  # pip install fairscale, fairscale-0.3.2
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

torch.set_num_threads(1)

def get_parse_args() -> (argparse, object, object, object):
    
    model = 'transformer' # 'mix_LM', 'transformer', 'gMLP'
    task = 'SOP'  # 'ICT', 'SOP'

    parser = argparse.ArgumentParser()

    if model == 'mix_LM':
        from models.mix_LM import Config
        parser.add_argument("--config_file", default='models/mix_LM_config.json', type=str, help="")
        if task == 'ICT':
            from utils.pretrain_utils_ict import LMDataset
            from models.mix_LM import BertMultiTask_ICT as BertMultiTask
        else:
            from utils.pretrain_utils_sop import LMDataset
            from models.mix_LM import BertMultiTask_SOP as BertMultiTask
    elif model == 'gMLP':
        from models.gMLP import Config
        parser.add_argument("--config_file", default='models/gMLP_config.json', type=str, help="")
        if task == 'ICT':
            from utils.pretrain_utils_ict import LMDataset
            from models.gMLP import BertMultiTask_ICT as BertMultiTask
        else:
            from utils.pretrain_utils_sop import LMDataset
            from models.gMLP import BertMultiTask_SOP as BertMultiTask
    else:
        from models.transformer import Config
        parser.add_argument("--config_file", default='models/transformer_config.json', type=str, help="")
        if task == 'ICT':
            from utils.pretrain_utils_ict import LMDataset
            from models.transformer import BertMultiTask_ICT as BertMultiTask
        else:
            from utils.pretrain_utils_sop import LMDataset
            from models.transformer import BertMultiTask_SOP as BertMultiTask

    ## Required parameters
    parser.add_argument("--model_name", type=str, required=True, help="모델 이름 지정")

    ## File related parameters
    parser.add_argument("--corpus_path", default='data', type=str,
                        help="학습 데이터 경로 지정, 해당 경로에 .dat의 확장자를 갖는 파일을 넣으시오.")

    parser.add_argument("--output_dir", default='checkpoint', type=str,
                        help="학습 완료 후 모델이 저장될 경로 지정")
    parser.add_argument("--checkpoint", default=None, type=str, # 'checkpoint/hong_015_50000.bin'
                        help="사전 학습된 모델을 불러올지 여부 지정, length 128 사전 학습 모델을 불러올때 사용, 없으면 None")

    ## hyperparameter
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="토크나이저로 토큰화 후 최대 입력 시퀀시 길이 지정")
    parser.add_argument("--train_batch_size", default=0, type=int,
                        help="GPU 하나 당 배치 사이즈 지정, 0인 경우 자동으로 배치사이즈 조정")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="백워드 전에 누적할 업데이트 스탭 수를 지정")
    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="학습률 지정")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="가중치 감쇠율 지정")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam 옵티마이저 epsilon 값 지정")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm 값 지정")
    parser.add_argument('--seed', type=int, default=42,
                        help="초기화를 위한 랜덤 시드 값 지정")

    ## Learning procedure parameters
    parser.add_argument("--num_steps", default=400000, type=int,
                        help="총 학습 스탭 수 지정")
    parser.add_argument("--scheduler_num_steps", default=500000, type=int,
                        help="스케쥴러의 최종 스탭 수 지정, decay가 해당 지점까지 내려감, num_steps보다 더 크게 잡아서 중간에 끝나도록 할 수 있음")
    parser.add_argument("--max_steps_per_epoch", default=1000, type=int,
                        help="에포크 당 최대 스탭, 데이터의 길이를 맞추는 역할")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="웜업 스탭 수 지정")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="로그 값을 저장할 스탭 수 지정")
    parser.add_argument('--save_steps', type=int, default=50000,
                        help="모델을 저장할 스탭 수 지정")

    ## AUTOMATIC MIXED PRECISION
    parser.add_argument('--fp16', action='store_true',
                        help="pytorch 내부 Automatic Mixed Precision 사용 여부 지정")

    return parser.parse_args(), BertMultiTask, Config, LMDataset


def get_file_list(args) -> list:
    file_list = os.listdir(args.corpus_path)
    data_file = [file for file in file_list if file.endswith(".dat")]
    
    return data_file

def get_logger() -> logging:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    return logging.getLogger(__name__)

def init_Q(args, queue):
    def creator(data, q):
        random.shuffle(data)

        for item in data:
            q.put(item)

    max_len_loop = ((args.num_steps * args.gradient_accumulation_steps) // args.max_steps_per_epoch) // len(args.data_file) + 10
    for i in range(max_len_loop):
        creator(args.data_file, queue)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(ARGS):
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

def training_step(model: torch.nn.Module, batch: list, args: object) -> torch.Tensor:
    
    model.train()
    model.zero_grad()
    
    batch = tuple(t.to(args.device) for t in batch)
    input_ids, input_mask, segment_ids, lm_label_ids, sentence_order_label = batch
    
    if args.fp16:
        with torch.cuda.amp.autocast():
            mlm_loss, aux_loss = model(
                input_ids = input_ids, 
                token_type_ids = segment_ids, 
                attention_mask = input_mask,
                masked_lm_labels = lm_label_ids,
                next_sentence_label = sentence_order_label
            )
    else:
        mlm_loss, aux_loss = model(
            input_ids = input_ids, 
            token_type_ids = segment_ids, 
            attention_mask = input_mask,
            masked_lm_labels = lm_label_ids,
            next_sentence_label = sentence_order_label
        )
    
    loss = mlm_loss + aux_loss
    
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
        
    if args.fp16:
        model.scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.detach(), mlm_loss.detach(), aux_loss.detach()


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
    
    model = BertMultiTask(args.config, train_mode='pretrain')

    if args.checkpoint != None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        
    model.to(args.device)

    if args.fp16:
        model.scaler = ShardedGradScaler()

    return model


def prepare_optimizer(model: object, args: object) -> (object, object):
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    base_optimizer_arguments = {
        "lr": args.learning_rate,
        'eps': args.adam_epsilon
    }
    base_optimizer = AdamW  
    optimizer = OSS(
        params = optimizer_grouped_parameters,
        optim = base_optimizer,
        **base_optimizer_arguments
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.scheduler_num_steps
    )
    
    return optimizer, scheduler

    
def adaptive_batch_size(
    rank: int,
    world_size: int,
    args: object):
    global Q
    
    # process group init
    torch.cuda.set_device(rank)
    args.device = torch.device("cuda", rank)
    args.n_gpu = 1
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:{}".format(RPC_PORT), rank=rank, world_size=world_size)
    
    model = prepare_model(args)
    optimizer, _ = prepare_optimizer(model, args)
    
    model = ShardedDDP(model, optimizer)
    
    loading_file = args.corpus_path + '/' + Q.get()
    max_len = 9999999
    train_dataset = LMDataset(corpus_path=loading_file,
                          tokenizer=args.tokenizer,
                          local_rank=rank,
                          max_len=max_len,
                          seq_len=args.max_seq_length,
                          vocab_size=args.vocab_size)
    
    adapt = True  # while this is true, the algorithm will perform batch adaptation
    time_step = 5
    old_batch_size = 0
    gpu_batch_size = 5  # initial gpu batch_size, it can be super small
    continue_training = True
    
    # Modified training loop to allow for adaptive batch size
    while continue_training:
        
        if rank == 0: print("Batch Size :", gpu_batch_size)
        train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=int(gpu_batch_size),
                    drop_last=True
                )
        
        try:
            for step, batch in enumerate(train_dataloader):
                
                if (step + 1) % args.gradient_accumulation_steps != 0:
                    with model.no_sync():
                        loss, mlm_loss, aux_loss = training_step(model, batch, args)
                else:
                    loss, mlm_loss, aux_loss = training_step(model, batch, args)
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    
                    optimizer.step()

                    if adapt and step > 2:
                        old_batch_size = gpu_batch_size
                        gpu_batch_size += time_step
                        break
                        
                    if step > gpu_batch_size:
                        continue_training = False
                        
                    torch.cuda.empty_cache()
                    
        except RuntimeError as run_error:
            
            gpu_batch_size = old_batch_size
            adapt = False  # turn off the batch adaptation
            continue_training = False
            break
        
    gpu_batch_size *= args.gradient_accumulation_steps
    file = 'batch.config'
    if os.path.isfile(file):
        with open(file, "r", encoding='utf-8') as config_file:
            batch_size = int(config_file.readlines()[0])
            
        if batch_size < gpu_batch_size:
            gpu_batch_size = batch_size
        
    with open(file, "w", encoding='utf-8') as config_file:
        config_file.write(str(gpu_batch_size))
    
    
def train(
    rank: int,
    world_size: int,
    args: object):
    global Q

    if rank == 0:
        tb_dir = os.path.join("tensorboard", args.model_name)
        os.makedirs(args.output_dir, exist_ok=True)  # 체크포인트 디렉토리 체크
        os.makedirs(tb_dir, exist_ok=True)  # 텐서보드 디렉토리 체크
        tb_writer = SummaryWriter(tb_dir)
    
    # process group init
    torch.cuda.set_device(rank)
    args.device = torch.device("cuda", rank)
    args.n_gpu = 1
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:{}".format(RPC_PORT), rank=rank, world_size=world_size)
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # model init
    model = prepare_model(args)
    optimizer, scheduler = prepare_optimizer(model, args)
    
    model = ShardedDDP(model, optimizer)
    
    if rank == 0:
        LOGGER.info("Model Loading : %s", args.checkpoint)
        num_params = count_parameters(model.module)
        model_params_encoder = count_parameters(model.module.bert.encoder)
        LOGGER.info("Total Parameter: %d" % num_params)
        LOGGER.info("Model Parameter: %d" % (model_params_encoder))
        LOGGER.info("FP16 Enable : %s", args.fp16)
        LOGGER.info("***** Running training *****")
        LOGGER.info("  Total optimization steps = %d", args.num_steps)
        LOGGER.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        LOGGER.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    iter = 0
    while True:
        
        train_dataloader, max_len = prepare_dataloader(rank, Q, args)
        
        if rank == 0:
            LOGGER.info("Train Dataset Len : %s", max_len)
        
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Training (X iter) (XX / XX Steps) (total_loss=X.X) (MLM loss=X.X) (AUX loss=X.X)",
            disable=rank != 0)
        
        tr_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            
            loss, mlm_loss, aux_loss = training_step(model, batch, args)
            tr_loss += loss
                
            mean_loss = tr_loss * args.gradient_accumulation_steps / (step+1)

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
                    "Training (%d iter) (%d / %d Steps) (Total loss=%2.5f) (MLM loss=%2.5f) (AUX loss=%2.5f)"
                    % (iter, global_step, args.num_steps, mean_loss, mlm_loss, aux_loss))

                if rank == 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('Total_Loss', mean_loss, global_step)
                    tb_writer.add_scalar('First_Loss', mlm_loss, global_step)
                    tb_writer.add_scalar('Second_Loss', aux_loss, global_step)
                    
                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
                    torch.save(model_to_save.state_dict(), model_checkpoint)
                    LOGGER.info("Saving model checkpoint to %s", args.output_dir)
                
            if args.num_steps > 0 and global_step == args.num_steps:
                break
                
        epoch_iterator.close()
        del train_dataloader
        del epoch_iterator
        gc.collect()
        
        if args.num_steps > 0 and global_step == args.num_steps:
            break
            
        iter += 1
        
    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
        torch.save(model_to_save.state_dict(), model_checkpoint)
        LOGGER.info("Saving model checkpoint to %s", args.output_dir)
        LOGGER.info("End Training!")
        tb_writer.close()

def main():
    global ARGS

    # Set seed
    set_seed(ARGS)
    
    # Set Configuration
    ARGS.config = Config(ARGS.config_file)
    ARGS.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    ARGS.vocab_size = ARGS.tokenizer.vocab_size
    ARGS.config.max_seq_length = ARGS.max_seq_length
    
    if ARGS.train_batch_size == 0:
        
        file = 'batch.config'
        if os.path.isfile(file):
            os.remove(file)
            
        # adaptive batch size
        mp.spawn(
            adaptive_batch_size,
            args=(ARGS.world_size, ARGS),
            nprocs=ARGS.world_size,
            join=True
        )

        with open(file, "r", encoding='utf-8') as config_file:
            batch_size = int(config_file.readlines()[0])
        ARGS.train_batch_size = batch_size - max(batch_size // 9, 5)

        os.remove(file)
        LOGGER.info("train_batch_size : %s", ARGS.train_batch_size)
    
    # Training
    mp.spawn(
        train,
        args=(ARGS.world_size, ARGS),
        nprocs=ARGS.world_size,
        join=True
    )

ARGS, BertMultiTask, Config, LMDataset = get_parse_args()
ARGS.data_file = get_file_list(ARGS)
ARGS.world_size = max(torch.cuda.device_count(), 1)
LOGGER = get_logger()
RPC_PORT = 29501
Q = mp.SimpleQueue()
init_Q(ARGS, Q)
    
if __name__ == "__main__":
    main()