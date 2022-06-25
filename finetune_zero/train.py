# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadV1Processor

from models.mix_LM import Config, BertMultiTask_ICT as BertMultiTask

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
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    optimizer = AdamW(params=optimizer_grouped_parameters,
                  lr=args.learning_rate,
                  eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=t_total * 0.1,
        num_training_steps=t_total
    )
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    
    model.zero_grad()
    model.train()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XXXX/XXXX) (loss=XX.XXXXX)")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)  # multi-gpu does scattering it-self
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

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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
                    "Train(%d Epoch) Step(%d / %d) (loss=%5.5f)" % (_, global_step, t_total, loss))

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

def load_and_cache_examples(args, tokenizer):

    # Load data features from cache or dataset file
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}'.format('train',
        str(args.max_seq_length)))
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default='pretrain_ckpt/hong_018_50.bin', type=str,
                        help="불러올 사전 학습된 모델의 경로 및 파일명 지정")
    parser.add_argument("--output_dir", default='checkpoint', type=str,
                        help="학습 완료 후 모델이 저장될 경로 지정")

    ## Other parameters
    parser.add_argument("--train_file", default='data/KorQuAD_v1.0_train.json', type=str,
                        help="학습 데이터셋의 경로 및 파일명 지정")
    parser.add_argument("--config_name", default="models/mix_LM_config.json", type=str,
                        help="사전 학습된 모델의 경로 및 파일명 지정")


    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="토크나이저로 토큰화 후 최대 입력 시퀀시 길이 지정")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="문서의 길이가 max_seq_length보다 길때 나누어 입력으로 지정함, 이때 얼마나 건너뛰어 나눌지 지정")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="질문의 최대 길이 지정, 값을 넘는 질문 토큰을 자름")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="gpu 당 배치 사이즈 지정.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="학습률 지정")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="백워드 전에 누적할 업데이트 스탭 수를 지정")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="가중치 감쇠율 지정")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Adam 옵티마이저 epsilon 값 지정")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm 값 지정")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="총 학습 에포크 수 지정")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="nbest_predictions.json 출력 파일에서 생성할 총 n-best 예측 수 지정")
    parser.add_argument("--max_answer_length", default=20, type=int,
                        help="생성 할 수있는 답변의 최대 길이 지정, 시작 및 종료 예측이 서로 조건화되지 않기 때문에 필요함")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="초기화를 위한 랜덤 시드 값 지정")

    parser.add_argument('--fp16', action='store_true',
                        help="pytorch 내부 Automatic Mixed Precision 사용 여부 지정")
    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s, 16-bits training: %s",
                   device, args.n_gpu, args.fp16)
    
    args.config = Config.from_json_file(args.config_name)
    args.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    args.vocab_size = args.tokenizer.vocab_size
    args.config.vocab_size = args.vocab_size
    args.config.max_seq_length = args.max_seq_length
    
    model = BertMultiTask(args.config, train_mode='mrc')
    model.load_state_dict(torch.load(args.checkpoint))
    
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, args.tokenizer)
    train(args, train_dataset, model)

if __name__ == "__main__":
    main()
