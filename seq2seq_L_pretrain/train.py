# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


from models.seq_seq import Config, Transformer
from utils.pretrain_utils_seq import LMDataset

logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model):

    tb_dir = os.path.join("tensorboard", args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)  # 체크포인트 디렉토리 체크
    os.makedirs(tb_dir, exist_ok=True)  # 텐서보드 디렉토리 체크
    tb_writer = SummaryWriter(tb_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.warmup_steps == 0:
        logger.info('WARNING!!! : warmup_step is 0')
    
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
        num_training_steps=args.num_steps
    )
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    iter = 0
    while True:
        
        train_dataset = LMDataset(corpus_path=args.corpus_path,
                                  tokenizer=args.tokenizer,
                                  local_rank=-1,
                                  seq_len=args.max_seq_length,
                                  vocab_size=args.vocab_size)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader,
                              desc="Training (X iter) (XX / XX Steps) (total_loss=X.X)\
                               (MLM loss=X.X) (AUX loss=X.X)")
        
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            
            batch = tuple(t.to(args.device) for t in batch)
            src_ids, tgt_ids, src_mask, tgt_mask = batch
            
            if args.fp16:
                with torch.cuda.amp.autocast():
                    loss = model(src_ids, tgt_ids[:,:-1], src_mask, tgt_mask, tgt_ids[:, 1:])
            else:
                loss = model(src_ids, tgt_ids[:,:-1], src_mask, tgt_mask, tgt_ids[:, 1:])

            if args.n_gpu > 1: # mean() to average on multi-gpu parallel training
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            mean_loss = tr_loss * args.gradient_accumulation_steps / (step+1)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    skip_lr_sched = (scale != scaler.get_scale())
                    if not skip_lr_sched:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d iter) (%d / %d Steps) (Total loss=%2.5f)"
                    % (iter, global_step, args.num_steps, mean_loss))

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('Total_Loss', mean_loss, global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
                    torch.save(model_to_save.state_dict(), model_checkpoint)
                    logger.info("Saving model checkpoint to %s", args.output_dir)

            if args.num_steps > 0 and global_step == args.num_steps:
                epoch_iterator.close()
                break
        if args.num_steps > 0 and global_step == args.num_steps:
            epoch_iterator.close()
            break
        
        epoch_iterator.close()
        iter += 1

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_checkpoint = os.path.join(args.output_dir, args.model_name+'_'+str(global_step)+'.bin')
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saving model checkpoint to %s", args.output_dir)
    logger.info("End Training!")
    tb_writer.close()



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", default='models/seq_seq_config.json', type=str, 
                        help="모델 설정 파일 지정")

    ## Required parameters
    parser.add_argument("--model_name", type=str, required=True, help="모델 이름 지정")
    
    ## File related parameters
    parser.add_argument("--corpus_path", default='data/pretrain_data.dat', type=str,
                        help="학습 데이터 파일 지정")
    parser.add_argument("--output_dir", default='checkpoint', type=str,
                        help="학습 완료 후 모델이 저장될 경로 지정")
    
    ## hyperparameter
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="토크나이저로 토큰화 후 최대 입력 시퀀시 길이 지정")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="전체 배치 사이즈 지정")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="백워드 전에 누적할 업데이트 스탭 수를 지정")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
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
    parser.add_argument("--num_steps", default=500000, type=int,
                        help="총 학습 스탭 수 지정")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="웜업 스탭 수 지정")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="로그 값을 저장할 스탭 수 지정")
    parser.add_argument('--save_steps', type=int, default=50000,
                        help="모델을 저장할 스탭 수 지정")
    

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s, 16-bits training: %s",
                   device, args.n_gpu, args.fp16)
    
    args.config = Config(args.config_file)
    args.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    special_tokens_dict = {'additional_special_tokens': ['[EOS]']}
    args.tokenizer.add_special_tokens(special_tokens_dict)
    args.vocab_size = args.config.vocab_size
    args.config.max_seq_length = args.max_seq_length
    
    model = Transformer(args.config)
    
    num_params = count_parameters(model)
    model_params_encoder = count_parameters(model.encoder) + count_parameters(model.decoder)
    logger.info("{}".format(args.config))
    logger.info("Total Parameter: %d" % num_params)
    logger.info("Model Parameter: %d" % (model_params_encoder))

    model.to(args.device)
    
    logger.info("Training parameters %s", args)

    # Training
    train(args, model)

if __name__ == "__main__":
    main()
