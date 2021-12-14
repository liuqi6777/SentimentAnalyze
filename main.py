# -*- coding: utf-8 -*-
# @Author : Liu Qi

import argparse
import logging
import os

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from datautils import load_dataset, build_dataloader
import config


class Model:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.model = opt.model(self.opt)
        self.optimizer = opt.optimizer(self.model.parameters(), lr=self.opt.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data_loader, val_data_loader=None):
        losses = []
        for epoch in tqdm(range(self.opt.epoch)):
            loss_list = []
            for batch, (input, target) in enumerate(train_data_loader):
                self.optimizer.zero_grad()
                output, _ = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                print('epoch {}: batch {} loss: {}'.format(epoch+1, batch, loss.item()), end='\r')

                loss_list.append(loss.item())

            epoch_loss = sum(loss_list) / len(loss_list)
            print("[INFO] average loss of epoch {}: {}".format(epoch+1, epoch_loss))
            losses.append(epoch_loss)

    def evaluate(self, test_data_loader):
        n_total = 0
        outputs_all = torch.tensor([])
        targets_all = torch.tensor([])
        with torch.no_grad():
            for batch, (input, target) in enumerate(test_data_loader):
                output, _ = self.model(input)
                output = torch.argmax(output, dim=-1)
                n_total += len(output)

                targets_all = torch.cat((targets_all, target), dim=0)
                outputs_all = torch.cat((outputs_all, output), dim=0)

        def cal_accuracy(target, output):
            return ((output == target).sum() / len(output)).item()

        def cal_precision(target, output):
            return (((target == 1) & (output == 1)).sum() / (output == 1).sum()).item()

        def cal_recall(target, output):
            return (((target == 1) & (output == 1)).sum() / (target == 1).sum()).item()

        def cal_f1_score(target, output):
            return (2 / (1 / cal_precision(target, output) + 1 / cal_recall(target, output)))

        acc = cal_accuracy(targets_all, outputs_all)
        prc = cal_precision(targets_all, outputs_all)
        rec = cal_recall(targets_all, outputs_all)
        f1 = cal_f1_score(targets_all, outputs_all)

        print('accuracy  : {}'.format(acc))
        print('precision : {}'.format(prc))
        print('recall    : {}'.format(rec))
        print('f1-score  : {}'.format(f1))

        return acc, prc, rec, f1

    def save_model(self, save_path, epoch):
        print('[INFO] Saving to {}_{}.pth'.format(save_path, epoch))
        torch.save(self.model.state_dict(), '{}_{}.pth'.format(save_path, epoch))

    def load_model(self, save_path, epoch):
        print('[INFO] Loading from {}_{}.pth'.format(save_path, epoch))
        self.model.load_state_dict(torch.load('{}_{}.pth'.format(save_path, epoch)))

    def run(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lstm', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--embeddings', default='random', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--output_size', default=2, type=int)
    parser.add_argument('--max_seq_len', default=30, type=int)
    parser.add_argument('--dataset', default='rt-polarity', type=str)
    opt = parser.parse_args()

    opt.model = config.model_classes[opt.model]
    opt.optimizer = config.optimizers[opt.optimizer]
    opt.pretrain = config.pretrains[opt.embeddings]
    opt.vocab_size = 30000

    model = Model(opt)

    train_set = load_dataset('data/train.pkl')
    test_set = load_dataset('data/test.pkl')
    train_loader = build_dataloader(train_set, batch_size=256)
    test_loader = build_dataloader(test_set, batch_size=256)

    model.train(train_loader)
    model.evaluate(test_loader)


if __name__ == '__main__':
    main()
