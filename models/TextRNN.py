from pickle import NONE
from typing import Tuple
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.embedding_dim = opt.embedding_dim
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers

        if opt.pretrain is None:
            self.embedding = nn.Embedding(opt.vocab_size, self.embedding_dim)
        else:
            pass
            # self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False)
        self.output = nn.Linear(self.hidden_size, self.output_size)

        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x: Tensor, hidden: Tensor=None):
        if hidden is None:
            hidden = x.data.new(self.num_layers, x.shape[0], self.hidden_size).fill_(0).float()
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.output(output[:, -1, :])
        return output, hidden
