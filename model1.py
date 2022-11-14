import random
import torch
from torch import optim
from torch.distributions import Bernoulli
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import numpy
from torch.utils.data import DataLoader


class ModelSimple(nn.Module):
    def __init__(self, len_data):
        super().__init__()
        self.msg_len = 4
        self.lstm_out = 100
        self.base_inp = len_data
        self.activation = nn.Tanh()
        self.linear = nn.Linear(self.lstm_out, self.msg_len + 1)
        self.num_loop = 2
        self.num_layers = 1
        self.lstm = nn.LSTM(self.base_inp + self.msg_len, self.lstm_out, num_layers=self.num_layers)

    @staticmethod
    def cat(b, msg):
        return torch.cat([b.unsqueeze(0), msg], dim=-1)

    @staticmethod
    def stack(b, msg):
        pass

    def forward(self, batch):
        len_batch = len(batch)
        self.h1 = torch.zeros(self.num_layers, len_batch, self.lstm_out).to(batch)
        self.c1 = torch.zeros(self.num_layers, len_batch, self.lstm_out).to(batch)
        self.h2 = torch.zeros(self.num_layers, len_batch, self.lstm_out).to(batch)
        self.c2 = torch.zeros(self.num_layers, len_batch, self.lstm_out).to(batch)

        msg1 = torch.zeros(1, len(batch), self.msg_len).to(batch)
        msg2 = torch.zeros(1, len(batch), self.msg_len).to(batch)

        # we need unique hidden state
        # because inputs are unique
        out1, (self.h1, self.c1) = self.lstm(self.cat(batch[:, :self.base_inp], msg1), (self.h1, self.c1))
        out1 = self.activation(self.linear(out1))
        out2, (self.h2, self.c2) = self.lstm(self.cat(batch[:, self.base_inp:], msg2), (self.h2, self.c1))

        out2 = self.activation(self.linear(out2))
        for i in range(self.num_loop):
            msg1 = out1[:, :, :self.msg_len]
            msg2 = out2[:, :, :self.msg_len]
            out1, (self.h1, self.c1) = self.lstm(self.cat(batch[:, :self.base_inp], msg1), (self.h1, self.c1))
            out1 = self.activation(self.linear(out1))
            out2, (self.h2, self.c1) = self.lstm(self.cat(batch[:, self.base_inp:], msg2), (self.h2, self.c1))
            out2 = self.activation(self.linear(out2))
        # out is tanh
        res1 = Bernoulli((out1[:, :, -1] + 1)/2)
        res2 = Bernoulli((out2[:, :, -1] + 1)/2)
        return res1, res2
