import torch

from models import RNN, LSTM, GRU


model_classes = {
    'rnn': RNN,
    'lstm': LSTM,
    'gru': GRU
}

model_hyper_params = {
    'rnn': {
        'vocab_size': 0,
        'embedding_dim': 0,
        'hidden_size': 0,
        'output_size': 0,
        'num_layers': 0,
        'dropout': 0
    },
    'lstm': {
        'vocab_size': 0,
        'embedding_dim': 0,
        'hidden_size': 0,
        'output_size': 0,
        'num_layers': 0,
        'dropout': 0
    },
    'gru': {
        'vocab_size': 0,
        'embedding_dim': 0,
        'hidden_size': 0,
        'output_size': 0,
        'num_layers': 0,
        'dropout': 0
    }
}

vocab_size = 0

batch_size = 256

optimizers = {
    'adam': torch.optim.Adam,  # default lr=0.001
    'sgd': torch.optim.SGD,
}

lr = 1e-3

epochs = 20

datasets = {
    'rt-polarity': {
        'train': 'data/train.pkl',
        'test': 'data/test.pkl'
    }
}

pretrains = {
    'random': None,
    'glove.6B.50d': ''
}