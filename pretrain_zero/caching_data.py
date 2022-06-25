import transformers
from utils.pretrain_utils_sop import LMDataset

tokenizer = transformers.ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

import gc
import os
def get_file_list(corpus_path) -> list:
    file_list = os.listdir(corpus_path)
    data_file = [file for file in file_list if file.endswith(".dat")]
    
    return data_file
    
corpus_path = 'data'
data_file = get_file_list(corpus_path)
print(data_file)

for i in data_file:
    print(i)
    loading_file = corpus_path + '/' + i
    train_dataset = LMDataset(corpus_path=loading_file,
                              tokenizer=tokenizer,
                              local_rank=-1,
                              seq_len=512,
                              vocab_size=35000)
    del train_dataset
    gc.collect()

print('캐싱 완료')