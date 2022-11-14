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
from model1 import ModelSimple


def get_data(len_data):
    same = bool(random.randint(0, 1))
    s1 = random.randint(0, 2 ** len_data)
    s2 = random.randint(0, 2 ** len_data)
    if same:
        s2 = s1
    ar1 = numpy.zeros(len_data + 1, dtype=numpy.uint8)
    ar2 = numpy.zeros(len_data + 1, dtype=numpy.uint8)
    for i in range(len_data + 1):
        ar1[i] = int(bool(s1 & (1 << i)))
        ar2[i] = int(bool(s2 & (1 << i)))
    items = torch.tensor(ar1), torch.tensor(ar2)
    if all(items[0] == items[1]):
         expect = 1
    else:
         expect = 0
    return torch.cat(items), expect


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, p):
        super().__init__()
        self._len = 10000
        self._pow = p

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return get_data(self._pow)


def main():
    epochs = 150
    episode_len = 1000
    batch_size = 100
    len_data = 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModelSimple(len_data + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = DigitDataset(len_data)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    for i in range(epochs):
        rewards = []
        for batch in loader:
            model.zero_grad()
            optimizer.zero_grad()
            batch, expect = batch
            expect = expect.to(device)
            d1, d2 = model(batch.float().to(device))
            a1 = d1.sample()
            a2 = d2.sample()
            reward = torch.logical_and(a1 == expect, a2 == expect)
            reward = reward.float() - 1 * torch.logical_not(reward).float()
            # loss = -d1.log_prob(a1) * reward - d2.log_prob(a2) * reward
            # loss = -d1.log_prob(a1) * (a1 == expect) - d2.log_prob(a2) * (a2 == expect)
            loss = -d1.log_prob(a1) * (a1 == expect)
            loss.mean().backward()
            optimizer.step()
            rewards.append((reward > 0).float().mean())
        print(torch.tensor(rewards).mean())

if __name__ == '__main__':
    main()

