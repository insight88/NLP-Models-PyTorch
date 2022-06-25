# pretrain : 사전학습을 위한 언어 모델 소스 코드
 - fairscale을 통해 사전학습을 위한 코드
   1. train_zero_one.py : ZeRO 1 학습을 위한 코드
   2. train_zero_two.py : ZeRO 2와 ZeRO 3 학습을 위한 코드
   3. train : 일반 학습을 위한 코드 
 - 실행 방법은 아래와 같이 학습 방법에 맞게 arguments를 같이 넣어 실행하면 됩니다.
 - 예시 > python train_zero_one.py --model_name hong <뒤로 arguments를 계속 입력>

### 1. 모델 설명 : 학습할 모델에 대한 설명, train 파일들 상단 get_parse_args() 함수에서 model과 task의 설정에 따라 각각 다른 모델이 작동함
 - Mix LM : Mix Net의 컨볼루션의 다른 커널을 갖는 그룹별 연산을 통해 정보 검색의 성능을 높인 모델
 - Transformer : switch transformer를 기반으로 한 정보 검색을 위한 소스 오픈 가능 모델
 - SOP와 ICT : 사전학습 시 MLM 작업과 함께 SOP와 ICT 중 어떤 작업으로 사전학습할지를 결정함

### 2. 작업 설명 : 모델 선언시 train_mode를 아래 설명한 작업으로 입력하면 작동함
 - pretrain : 사전학습 전용 작업, MLM은 일부 토큰을 마스크하고 맞추도록 하는 작업, SOP는 두 문장의 순서가 맞는 지를 학습하는 작업, ICT는 두 문장의 관련성을 학습하는 작업
 - mrc : 기계 독해 파인튜닝 전용 작업, 문장과 질문을 주고 문장에서 질문에 관련된 답을 찾아주는 작업
 - ict : 정보 검색 파인튜닝 전용 작업, 문장 쌍에 대한 관련성을 학습하는 작업으로 배치로 확장하여 다른 배치 간의 관련성도 학습함
 - encoder : 버트에서 sequence_outputs를 뽑기 위한 함수로 precompute시나 오픈QA 때 문장을 인코딩하기 위한 함수, BERT와 decoder을 통과한 (배치, 피쳐크기)의 텐서가 결과로 도출됨

### 3. 하이퍼 파라미터 설명 : 학습 설정을 위한 하이퍼 파라미터 소개
 - model_name : 필수 입력 사항으로 log 값과 체크 포인트의 이름을 결정하는 인자
 - corpus_path : 사전학습을 위한 말뭉치 경로, 말뭉치 경로에 .dat 파일로 넣으면 토큰화한 cache 파일을 자동으로 생성함
 - output_dir : 사전학습 후 모델 가중치 결과 파일을 저장하는 경로
 - checkpoint : 미리 학습된 파일을 불러와 재학습해야 할경우 파일 지정, 디렉토리와 파일명을 직접 지정해야 함
 - max_seq_length : 최대 입력 가능한 토큰 개수, 값이 커질수록 필요 GPU 메모리 크기가 커짐, 특히 트랜스포머 계열은 주의 필요
 - train_batch_size : 학습을 위한 배치 사이즈 결정, GPU 당 개수임, 제로 트레이너는 0을 입력하면 자동으로 배치 사이즈를 계산함(별도 계산 과정이 나옴)
 - gradient_accumulation_steps : 학습 시 정해진 GPU 크기 내에서 배치 사이즈를 올리기 위한 기법으로 옵티마이저 업데이트를 정해진 배율만큼 뒤로 미룸
 - learning_rate : 학습률 설정, 스케쥴러가 갖는 최대 학습률을 기준으로 작성, 프리트레인 : 2e-4, 파인튜닝 : 5e-5
 - weight_decay : 가중치 감쇠율 지정
 - adam_epsilon : AdamW 옵티마지저의 입실론값 지정
 - max_grad_norm : Max gradient norm 값 지정
 - seed : 랜덤 시드 값 지정, 이전 가중치 불러와 학습할 경우 이전 가중치와 동일한 랜덤 시드 값을 사용해야 함
 - num_steps : 최대 학습 스탭수 지정, 학습시 배치 사이즈 * 최대 스탭을 통해 모델이 얼마나 많은 데이터를 많이 봤느냐가 중요함, 배치사이즈를 줄이면 스탭 수를 늘여 학습하는 방법도 있음
 - scheduler_num_steps : 스케쥴러의 전체 주기 스탭수 지정, 학습률 감쇠가 있는 스케쥴러의 경우 이 값을 크게 잡고, num_steps를 작게 잡으면 학습률이 0이 되기 전에 학습을 종료 시킬 수 있음
 - max_steps_per_epoch : 데이터를 스탭 당 얼마나 샘플링 할 지를 결정함, 배치 사이즈 * 이 값이 학습 데이터 수를 넘으면 안됨
 - warmup_steps : 워밍업 스탭 수 지정, 0에서부터 learning_rate에 지정한 값까지 설정한 스탭수까지 선형으로 학습률이 올라감
 - logging_steps : 텐서보드에 표출한 그래프를 위한 데이터 값을 찍을 스탭 수 지정
 - save_steps : 저장할 스탭 수 지정
 - fp16 : 자동 믹스드 프리시젼으로 학습함(fp16을 기본으로 하고 필요한 경우 자동으로 fp32로 연산함), pytorch 내장 Automatic Mixed Precision을 사용함, 실행시 "--fp16"을 입력해야 함
 - auto_wrap : fairscale의 오토랩 기능, 레이어별 래핑으로 연산 속도를 높임, 파라미터 공유가 있는 모델은 사용 자재, mix_LM 및 layer_group 설정이 파라미터 공유임, 실행시 "--auto_wrap"을 입력해야 함
 - zero3 : 모델의 GPU 메모리를 확보하기 위한 기술, 속도는 zero2가 약간 빠름, 켜져 있는 경우 zero3, 꺼져 있는 경우 zero2로 작동함, zero3로 실행시 "--zero3"을 입력해야 함

### 4. 모델 파라미터 설명 : models 폴더에 모델파일명_config.json 형태로 구성됨
 - act_fn : 활성화 함수 지정, gelu/relu/swish가 있음
 - dropout_prob : 드랍아웃 비율 지정, 추론시 model.eval을 하면 dropout이 자동으로 꺼짐
 - hidden_size : 히든 스테이트 사이즈 지정, ICT 모델에서는 (배치사이즈, 히든사이즈)로 벡터가 생성됨, 클수록 모델 수용력이 좋아짐
 - kernel_size : mix_LM의 컨볼루션의 최대 커널 사이즈 지정
 - n_experts : 히든 사이즈를 나누어 연산하는 데 그 개수 지정
 - initializer_range : weight의 초기화 값 범위 지정
 - ff_dim : 피드포워드의 확장 사이즈 지정
 - layer_group : 레이어를 몇개로 그룹화하여 그룹별로 파라미터를 공유할 지 지정, 12로 지정하면 알버트와 같고, 1로 지정하면 일반 버트임
 - num_heads : 멀티헤드 어텐션의 헤드 수 지정
 - max_position_embedding : 포지션 임베딩의 최대 사이즈 지정, 지정한 값을 넘어 max_seq_length를 지정하면 안됨
 - num_hidden_layers : 모델의 레이어 수 지정
 - type_vocab_size : 보캡 타입 지정
 - vocab_size : 보캡 사이즈 지정
 - LN_eps : layer normalization의 입실론 값 지정
 - ln_pos : 레이어의 각 블록에서 layer normalization의 연산 위치를 결정, 0이면 pre ln/1이면 post ln, 모델이 충분히 크면 pre가 좋고 작으면 post가 좋음
