import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.embedding_dim = opt.embedding_dim
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers

        self.embedding = nn.Embedding(opt.vocab_size, self.embedding_dim)
        self.output = nn.Linear(self.hidden_size, self.output_size)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            h_0 = x.data.new(self.num_layers, x.shape[0], self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, x.shape[0], self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        x = self.embedding(x)
        output, hidden = self.lstm(x, (h_0, c_0))
        output = self.output(output[:, -1, :])
        return output, hidden
