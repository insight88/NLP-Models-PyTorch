# NLP 모델

### pretrain_zero

* [ZeRO 기술](https://arxiv.org/abs/1910.02054)을 적용한 대용량 분산처리 사전 학습 언어 모델
* 활용 기술 : PyTorch, [fairscale](https://github.com/facebookresearch/fairscale)
* 활용 모델 : gMLP, Switch Transformer, Mix LM
* 훈련 Task : SOP, ICT
* 수행 Task : 기계독해 MRC, 정보검색 IR

### finetune_zero

* pretrain_zero로 사전 학습된 언어 모델의 파인튜닝 훈련을 위한 코드
* 활용 기술 : PyTorch, fairscale
