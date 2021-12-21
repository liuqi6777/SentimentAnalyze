import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.pretrain is None:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings=opt.pretrain, freeze=False)

        self.conv = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, opt.embedding_dim)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(opt.dropout)
        self.output = nn.Linear(256 * len((2, 3, 4)), opt.output_size)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        output = self.embedding(x)
        output = output.unsqueeze(1)
        output = torch.cat([self.conv_and_pool(output, conv) for conv in self.conv], 1)
        output = self.dropout(output)
        output = self.output(output)
        return output
