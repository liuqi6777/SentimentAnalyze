import torch

from models import RNN, LSTM, GRU, CNN


model_classes = {
    'rnn': RNN,
    'lstm': LSTM,
    'gru': GRU,
    'cnn': CNN,
}

optimizers = {
    'adam': torch.optim.Adam,  # default lr=0.001
    'sgd': torch.optim.SGD,
}

datasets = {
    'rt-polarity': {
        'train': 'data/train.pkl',
        'test': 'data/test.pkl'
    }
}
