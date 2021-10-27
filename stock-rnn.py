from __future__ import print_function

import argparse
import logging
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

DATA_URL = 'https://raw.githubusercontent.com/tonylaioffer/stock-prediction-lstm-using-keras/master/data/sandp500/all_stocks_5yr.csv'

def get_data(name, url=DATA_URL):
    df = pd.read_csv(url)
    df = df[df['Name'] == name]
    close_price = df.close.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    close_price = scaler.fit_transform(close_price).squeeze() # Scale the data

    return close_price


def process_data(data, lag):
    X, Y = [], []
    lag = lag
    for i in range(len(data) - lag - 1):
        X.append(data[i: (i + lag)])
        Y.append(data[(i + lag)])
    return np.array(X), np.array(Y)


class StockDataset(Dataset):
    def __init__(self, X, y, train=True, test_size=0.2):
        super(StockDataset, self).__init__()
        cutoff = int(test_size * len(X))
        if train:
            self.features = X[:cutoff]
            self.targets = y[:cutoff]
        else:
            self.features = X[cutoff:]
            self.targets = y[cutoff:]

        self.features = np.expand_dims(self.features, axis=1).astype(np.float32)
        self.targets = self.targets.astype(np.float32)

    def __len__(self):
        assert len(self.features) == len(self.targets)
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.targets[item]


class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size, dropout=0.1, bidirectional=True):
        super(RNN, self).__init__()
        self.num_directions = bidirectional + 1
        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)

        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, self.num_directions, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state


def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for features, targets in train_loader:
        features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        output, _ = model(features, None)
        loss = loss_fn(output.squeeze(), targets)
        loss.backward()
        optimizer.step()

    msg = "Train Epoch: {}\tloss={}".format(
        epoch, loss.item())
    logging.info(msg)


def test(args, model, device, test_loader, loss_fn, epoch):
    model.eval()
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            output, _ = model(features, None)
            test_loss = loss_fn(output.squeeze(), targets)

    if epoch % 20 == 0:
        logging.info("{{metricName: loss, metricValue: {}}}\n".format(test_loss))


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Stock price prediction with RNN")
    parser.add_argument("--company-name", type=str, default='AAPL', metavar='XXXX',
                        help="the name of the company for which the stock price is to be predicted (default: 'AAPL')")
    parser.add_argument("--lag", type=int, default=7, metavar="N",
                        help="the size of the lagging window to be used for forecasting")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=150, metavar="N",
                        help="number of epochs to train (default: 150)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")

    if dist.is_available():
        parser.add_argument("--backend", type=str, help="Distributed backend",
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    data = get_data(name=args.company_name)
    X, y = process_data(data, args.lag)

    train_dset = StockDataset(X, y, train=True)
    test_dset = StockDataset(X, y, train=False)

    train_loader = DataLoader(train_dset,
                              batch_size=len(train_dset),
                              shuffle=True,
                              **kwargs
                              )

    test_loader = DataLoader(test_dset,
                             batch_size=len(test_dset),
                             shuffle=True,
                             **kwargs,
                            )

    model = RNN(i_size=args.lag,
                h_size=64,
                n_layers=3,
                o_size=1,
                bidirectional=False,
                )
    model.to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, epoch)
        test(args, model, device, test_loader, loss_fn, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()