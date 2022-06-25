import random
import pickle

def main():
    with open('diag_src_kor', 'r', encoding='utf-8') as f:
        src = f.readlines()

    with open('diag_tgt_kor', 'r', encoding='utf-8') as f:
        tgt = f.readlines()

    datasets = []

    for i in range(len(src)):
        data = []
        src_string = src[i].strip()
        tgt_string = tgt[i].strip()

        if not (src_string and tgt_string):
            continue

        data.append(src_string)
        data.append(tgt_string)

        datasets.append(data)

    random.shuffle(datasets)
    print(len(datasets))
    val_dataset = datasets[:646]
    train_dataset = datasets[646:]

    with open('diag_train.pkl', 'wb') as writer:
        pickle.dump(train_dataset, writer, protocol=pickle.HIGHEST_PROTOCOL)

    with open('diag_val.pkl', 'wb') as writer:
        pickle.dump(val_dataset, writer, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

