# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import sys
import json
import random
import logging
import argparse
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

import transformers
from transformers.data.processors.squad import SquadV1Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

import fairscale  # pip install fairscale, fairscale-0.3.2
from fairscale.nn.wrap import auto_wrap
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.nn.data_parallel import FullyShardedDataParallel

from utils.mrc_utils import (RawResult, write_predictions)
from debug.evaluate_korquad import evaluate as korquad_eval
from models.mix_LM import Config, BertMultiTask_ICT as BertMultiTask

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(args, model, eval_examples, eval_features, tokenizer):
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model_chk = torch.load(args.checkpoint, map_location=torch.device(args.device))
    model.load_state_dict(model_chk)
    
    model.eval()
    all_results = []
    logger.info("Start evaluating!")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        with torch.no_grad():
            if args.zero:
                model.load_state_dict(model_chk)
            batch_start_logits, batch_end_logits = model(
                input_ids = input_ids,
                token_type_ids = segment_ids, 
                attention_mask = input_mask,
            )
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")

    write_predictions(eval_examples, eval_features, all_results,
                  args.n_best_size, args.max_answer_length,
                  False, output_prediction_file, output_nbest_file,
                  None, False, False, 0.0)
    
    expected_version = 'KorQuAD_v1.0'
    with open(args.predict_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            logger.info('Evaluation expects ' + expected_version +
                        ', but got dataset with ' + read_version,
                        file=sys.stderr)
        dataset = dataset_json['data']
    with open(os.path.join(args.output_dir, "predictions.json")) as prediction_file:
        predictions = json.load(prediction_file)
    logger.info(json.dumps(korquad_eval(dataset, predictions)))


def load_and_cache_examples(args, tokenizer):
    # Load data features from cache or dataset file
    input_file = args.predict_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format('dev',
        str(args.max_seq_length), args.doc_stride))
    
    processor = SquadV1Processor()
    input_a = input_file.split('/')
    examples = processor.get_train_examples(''.join(input_a[:-1]), filename=input_a[-1])
    
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        features, _ = transformers.squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=4,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return examples, features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default='checkpoint/pe_5e-05_64_5_0.bin', type=str,
                        help="불러올 학습된 모델의 경로 및 파일명 지정")
    parser.add_argument("--output_dir", default='debug', type=str,
                        help="학습 완료 후 predictions파일이 저장될 경로 지정")

    ## Other parameters
    parser.add_argument("--predict_file", default='data/KorQuAD_v1.0_dev.json', type=str,
                        help="예측 데이터셋의 경로 및 파일명 지정")
    parser.add_argument("--config_name", default="models/mix_LM_config.json", type=str,
                        help="사전 학습된 모델의 경로 및 파일명 지정")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="토크나이저로 토큰화 후 최대 입력 시퀀시 길이 지정")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="문서의 길이가 max_seq_length보다 길때 나누어 입력으로 지정함, 이때 얼마나 건너뛰어 나눌지 지정")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="질문의 최대 길이 지정, 값을 넘는 질문 토큰을 자름")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="gpu 당 배치 사이즈 지정.")
    parser.add_argument("--n_best_size", default=5, type=int,
                        help="nbest_predictions.json 출력 파일에서 생성할 총 n-best 예측 수 지정")
    parser.add_argument("--max_answer_length", default=20, type=int,
                        help="생성 할 수있는 답변의 최대 길이 지정, 시작 및 종료 예측이 서로 조건화되지 않기 때문에 필요함")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="초기화를 위한 랜덤 시드 값 지정")
    parser.add_argument('--zero', action='store_true',
                        help="만약 학습에서 zero2와 zero3로 학습했다면 이 기능을 켜세요. 배치마다 값이 변경되는 버그를 위한 조치")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args)
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    args.config = Config.from_json_file(args.config_name)
    args.tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    args.vocab_size = args.tokenizer.vocab_size
    args.config.vocab_size = args.vocab_size
    args.config.max_seq_length = args.max_seq_length
    
    model = BertMultiTask(args.config, train_mode='mrc')
    
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(args.device)

    logger.info("Evaluation parameters %s", args)

    # Evaluate
    examples, features = load_and_cache_examples(args, args.tokenizer)
    evaluate(args, model, examples, features, args.tokenizer)

if __name__ == "__main__":
    main()
