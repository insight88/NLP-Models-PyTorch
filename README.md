# NLP 모델

### pretrain_zero

* [ZeRO 기술](https://arxiv.org/abs/1910.02054)을 적용한 대용량 분산처리 사전 학습 언어 모델
* 활용 기술 : PyTorch, [fairscale](https://github.com/facebookresearch/fairscale)
* 활용 모델 : gMLP, Switch Transformer, Mix LM
* 훈련 Task : SOP, ICT
* 수행 Task : 기계독해 MRC, 정보검색 IR

### finetune_zero

* pretrain_zero로 사전 학습된 언어 모델의 파인튜닝 훈련 코드

### seq2seq_L_pretrain

* seq2seq, self-attention을 활용한 사전 학습 대형 언어 모델
* 활용 기술 : PyTorch, fairscale
* 수행 Task : 대화형 챗봇

### seq2seq_L_finetune

* seq2seq_L_pretrain로 사전 학습된 언어 모델의 파인튜닝 훈련 코드

### Open_QA

* 오픈도메인 질의응답을 위한 파인튜닝 훈련 코드
* 훈련 Task : ICT
* 수행 Task : 오픈도메인 QA
