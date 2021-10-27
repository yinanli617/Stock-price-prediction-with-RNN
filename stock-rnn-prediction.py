from __future__ import print_function

import argparse
import logging
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

DATA_URL = 'https://raw.githubusercontent.com/tonylaioffer/stock-prediction-lstm-using-keras/master/data/sandp500/all_stocks_5yr.csv'
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
TARGET_COL = 'close'

def get_data(name, url=DATA_URL):
    df = pd.read_csv(url)
    df = df[df['Name'] == name]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    targets = df[TARGET_COL].values
    features = df[FEATURE_COLS].values

    train_size = int(0.8 * len(features))
    X_train, X_test, y_train, y_test = features[:train_size],\
                                       features[train_size:],\
                                       targets[:train_size],\
                                       targets[train_size:]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).squeeze()

    return X_train, X_test, y_train, y_test, target_scaler


def process_data(features, targets, lag):
    X, Y = [], []
    lag = lag
    for i in range(len(features) - lag - 1):
        X.append(features[i: (i + lag)])
        Y.append(targets[(i + lag)])
    return np.array(X), np.array(Y)


class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size, dropout=0.1, bidirectional=True):
        super(RNN, self).__init__()
        self.num_directions = bidirectional + 1
        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)

        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out[:, -1, :] # get the last hidden state
        outs = self.out(r_out)

        return outs, hidden_state


def train(X_train, y_train, model, device, optimizer, loss_fn, epoch):
    model.train()
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    optimizer.zero_grad()
    output, _ = model(X_train, None)
    loss = loss_fn(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        msg = "Train Epoch: {}\tloss={}".format(
            epoch, loss.item())
        logging.info(msg)

    return loss.item()


def test(X_test, y_test, model, device, loss_fn, epoch, target_scaler):
    model.eval()
    with torch.no_grad():
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)
        output, _ = model(X_test, None)
        test_loss = loss_fn(output.squeeze(), y_test)

    y_test = y_test.detach().cpu().numpy().squeeze()
    outputs = output.detach().cpu().numpy().squeeze()
    MAPE = np.mean(np.abs(outputs - y_test) / np.abs(y_test))
    corr_coef = np.corrcoef(y_test, outputs)[0, 1]

    if epoch % 10 == 0:
        logging.info("{{metricName: log_loss, metricValue: {}}}".format(test_loss))
        logging.info("{{metricName: MAPE, metricValue: {}}}".format(MAPE))
        logging.info("{{metricName: corr_coef, metricValue: {}}}\n".format(corr_coef))

        plt.figure(figsize=(20, 8))
        plt.plot(target_scaler.inverse_transform(y_test.reshape(-1, 1)), label='ground_truth')
        plt.plot(target_scaler.inverse_transform(outputs.reshape(-1, 1)), label='predicted')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Stock price')
        plt.title(f'Epoch {epoch}')
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        plt.savefig(f'./plots/epoch_{epoch}.png')
        plt.close('all')


    return test_loss.item()


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Stock price prediction with RNN")
    parser.add_argument("--company-name", type=str, default='AAPL', metavar='XXXX',
                        help="the name of the company for which the stock price is to be predicted (default: 'AAPL')")
    parser.add_argument("--lag", type=int, default=14, metavar="N",
                        help="the size of the lagging window to be used for forecasting (default: 14)")
    parser.add_argument("--epochs", type=int, default=50, metavar="N",
                        help="number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
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
            level=logging.INFO)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.INFO,
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

    X_train, X_test, y_train, y_test, target_scaler = get_data(name=args.company_name)
    X_train, y_train = process_data(X_train, y_train, args.lag)
    X_test, y_test = process_data(X_test, y_test, args.lag)

    model = RNN(i_size=len(FEATURE_COLS),
                h_size=64,
                n_layers=3,
                o_size=1,
                bidirectional=False,
                )
    model.to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel
        model = Distributor(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train(X_train, y_train, model, device, optimizer, loss_fn, epoch)
        test(X_test, y_test, model, device, loss_fn, epoch, target_scaler)

    if (args.save_model):
        torch.save(model.state_dict(), "stock_rnn.pt")


if __name__ == "__main__":
    main()