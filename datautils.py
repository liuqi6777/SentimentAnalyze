import os
import pickle as pkl
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


MAX_VOCAB_SIZE = 10000
UNK = '<UNK>'
PAD = '<PAD>'

data_dir = 'data'
pos_data_path = 'rt-polarity.pos'
neg_data_path = 'rt-polarity.neg'
pretrain_path = 'glove.6B.50d.txt'
embedding_path = 'glove50d.npz'
vocab_path = 'vocab.pkl'
train_path = 'train.pkl'
test_path = 'test.pkl'


def build_vocab(file_path, vocab_dir, min_freq=0):
    contents = []
    if isinstance(file_path, str):
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.readlines()
    elif isinstance(file_path, (list, tuple)):
        for file in file_path:
            with open(file, 'r', encoding='utf-8') as f:
                contents += f.readlines()
    else:
        raise TypeError

    words = []
    for content in tqdm(contents):
        words += content.strip().split(' ')
    counter = Counter(words)
    words = [word for (word, freq) in counter.items() if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(words)}
    word2idx.update({UNK: len(word2idx), PAD: len(word2idx) + 1})
    pkl.dump(word2idx, open(vocab_dir, 'wb'))
    return word2idx


def build_pretrain(word2idx, pretrain_dir, embed_dim, save_embedding_dir=None):
    embeddings = np.zeros((len(word2idx), embed_dim))
    with open(pretrain_dir, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            lin = line.strip().split(" ")
            if lin[0] in word2idx:
                idx = word2idx[lin[0]]
                emb = [float(x) for x in lin[1:]]
                embeddings[idx] = np.array(emb, dtype='float32')
    if save_embedding_dir is not None:
        np.savez_compressed(save_embedding_dir, embeddings=embeddings)
    embeddings = torch.from_numpy(embeddings).to(torch.float32)
    return embeddings


def build_dataset(path, word2idx, pad_size, label=None):
    dataset = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            token = line.strip().split()
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
            words_idx = [word2idx.get(word, word2idx[UNK]) for word in token]
            dataset.append((words_idx, label) if label is not None else word2idx)
    return dataset


def load_dataset(path):
    return pkl.load(open(path, 'rb'))


def load_vocab(path):
    return pkl.load(open(path, 'rb'))


def build_dataloader(dataset, batch_size):
    x = np.zeros((len(dataset), len(dataset[0][0])))
    y = np.zeros((len(dataset)))
    for idx, (data, target) in enumerate(dataset):
        x[idx] = data
        y[idx] = target
    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(x, y)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


def main():
    pos_data = os.path.join(data_dir, pos_data_path)
    neg_data = os.path.join(data_dir, neg_data_path)
    pretrain = os.path.join(data_dir, pretrain_path)
    embedding = os.path.join(data_dir, embedding_path)
    train = os.path.join(data_dir, train_path)
    test = os.path.join(data_dir, test_path)

    word2idx = build_vocab((pos_data, neg_data), vocab_path, 2)

    build_pretrain(word2idx, pretrain, 50, embedding)

    pos_set = build_dataset(pos_data, word2idx, pad_size=32, label=1)
    neg_set = build_dataset(neg_data, word2idx, pad_size=32, label=0)

    train_set = pos_set[:-500] + neg_set[:-500]
    test_set = pos_set[-500:] + neg_set[-500:]

    pkl.dump(train_set, open(train, 'wb'))
    pkl.dump(test_set, open(test, 'wb'))


if __name__ == '__main__':
    main()
