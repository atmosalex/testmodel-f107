import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import datetime
import sys
import random
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as MSE
import mymodels

torch.manual_seed(99)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class F107_loader:
    def __init__(self, preprocessed_h5 = 'dataset.h5'):

        # load the F10.7 flux data from https://lasp.colorado.edu/lisird/data/penticton_radio_flux/:
        date_parser = lambda x: datetime.datetime.strptime(x, '%Y%m%d_%H%M%S.%f')
        ty = pd.read_csv('penticton_radio_flux.csv', parse_dates=['time (yyyyMMdd_HHmmss.SSS)'], date_parser=date_parser)
        # type is not datetime, it is numpy datetime, we can convert to epoch by casting to int to get nanoseconds:
        f107data_epoch = ty['time (yyyyMMdd_HHmmss.SSS)'].astype(int) / 10 ** 9
        f107data_y1 = ty['observed_flux (solar flux unit (SFU))']
        f107data_y2 = ty['adjusted_flux (solar flux unit (SFU))']

        # load the 30cm flux data from http://solar.nro.nao.ac.jp/norp/html/daily_flux.html:
        date_parser = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        ty = pd.read_csv('TYKW-NoRP_dailyflux.txt', skiprows=1, parse_dates=['Date'], date_parser=date_parser)
        # filter for only non-NaN values in both Date and 30cm column:
        ty = ty[~ty[['Date', '9.4 GHzss']].isnull().any(axis=1)]
        f30data_epoch = ty['Date'].astype(int) / 10 ** 9
        f30data_y = ty['9.4 GHzss']

        # get data availability time range to nearest second:
        epoch_first_nearestsecond = math.ceil(f107data_epoch.values[0])
        epoch_last_nearestsecond = int(f107data_epoch.values[-1])
        epoch_first_nearestsecond = max(math.ceil(f30data_epoch.values[0]), epoch_first_nearestsecond)
        epoch_last_nearestsecond = min(int(f30data_epoch.values[-1]), epoch_last_nearestsecond)

        # interpolate every 24 hrs between available date range:
        # create an array of integer epochs 24hrs apart
        self.daily_epoch = np.arange(epoch_first_nearestsecond, epoch_last_nearestsecond, 24 * 3600)
        # 10.7cm:
        self.series_daily_f107 = pd.Series(np.interp(self.daily_epoch, f107data_epoch, f107data_y2))
        self.f107 = self.series_daily_f107.values
        # 30cm:
        self.series_daily_f30 = pd.Series(np.interp(self.daily_epoch, f30data_epoch, f30data_y))
        self.f30 = self.series_daily_f30.values

        # scale:
        scale_data = True
        if scale_data:
            self.f107_m = self.f107.mean(0, keepdims=True)
            self.f107_s = self.f107.std(0, keepdims=True)
            self.f107 -= self.f107_m
            self.f107 /= self.f107_s
            self.f30_m = self.f30.mean(0, keepdims=True)
            self.f30_s = self.f30.std(0, keepdims=True)
            self.f30 -= self.f30_m
            self.f30 /= self.f30_s
        else:
            self.f107_s = 1
            self.f107_m = 0
            self.f30_s = 1
            self.f30_m = 0

        # the interpolation to every 24hrs ~halves the length of the dataset
        self.n_samples = self.daily_epoch.shape[0]

class F107data(Dataset):  # inherit the torch.utils.data.Dataset class
    def __init__(self, dataset, rollingtarget=True, seqlen=5):
        self.sl = seqlen


    def __getitem__(self, idx):
        # INPUTS/FEATURES:
        if idx - self.sl >= 0:
            f107inputs = self.f107[idx - self.sl: idx]
            f30inputs = self.f30[idx - self.sl: idx]
        else:
            # pad the time series with the value at index 0
            f107inputs = np.hstack((self.f107[0] * np.ones(self.sl - idx), self.f107[: idx]))
            f30inputs = np.hstack((self.f30[0] * np.ones(self.sl - idx), self.f30[: idx]))

        inputs = np.vstack((f107inputs, f30inputs)).T

        inputs = torch.tensor(inputs.astype(np.float32))  # make the second (1) dimension into lists

        # TARGETS:
        f107outputs = [self.f107[idx]]
        f30outputs = [self.f30[idx]]

        if self.rollingtarget:
            rollingf107 = np.hstack((f107inputs[1:], f107outputs))  # input rolled forward by input_lag
            rollingf30 = np.hstack((f30inputs[1:], f30outputs))
            target = np.vstack((rollingf107, rollingf30)).T
        else:
            target = np.array([f107outputs, f30outputs])

        target = torch.tensor(target.astype(np.float32))
        return inputs, target

    def __len__(self):
        return self.n_samples


def experiment_persistence(data, test_frac=0.2):
    # rescale:
    y = (data.f107 * dataset.f107_s) + dataset.f107_m
    # y = (data.f30 * dataset.f30_s) + dataset.f30_m

    test_idx = int(len(y) * (1 - test_frac))
    print(
        f"test period is {datetime.datetime.fromtimestamp(data.daily_epoch[test_idx])} to {datetime.datetime.fromtimestamp(data.daily_epoch[-1])}")
    fherror = []
    loss = nn.MSELoss()

    # use the entire dataset because this test is fast
    for fh in range(7):
        y_obs_all = []
        y_pred_all = []
        # walk forward validation across the testing data:
        for t in range(test_idx, len(y) - (fh + 1)):
            y_obs = y[t:t + (fh + 1)]  # get observational data across the forecast window
            y_pred = [y[t - 1]] * (fh + 1)  # previous value before forecast window
            # sigma = y_obs * 0.058 - 2.9
            y_obs_all.append(y_obs)
            y_pred_all.append(y_pred)
            # if datetime.datetime.fromtimestamp(data.daily_epoch[t]) > datetime.datetime(2015,1,1): break

        y_obs_all = np.array(y_obs_all)
        y_pred_all = np.array(y_pred_all)
        y_obs_all = torch.tensor(y_obs_all)
        y_pred_all = torch.tensor(y_pred_all)
        # error = math.sqrt(MSE(y_obs_all, y_pred_all))
        # error = math.sqrt(np.sum(np.power(y_obs_all - y_pred_all,2))/np.size(y_obs_all))
        fherror.append(np.sqrt(loss(y_pred_all, y_obs_all).item()))
        print(f'error with forecast horizon of {1 + fh} days = {fherror[-1]:.3f}')
    # error with forecast horizon of 1 days = 227.403
    # error with forecast horizon of 2 days = 244.351
    # error with forecast horizon of 3 days = 265.253
    # error with forecast horizon of 4 days = 288.846
    # error with forecast horizon of 5 days = 313.756
    # error with forecast horizon of 6 days = 339.794
    # error with forecast horizon of 7 days = 366.119


def experiment_ARIMA(data, test_frac=0.2):
    y = data.f107
    loss = nn.MSELoss()
    test_idx_first = int(len(y) * (1 - test_frac))
    autocorrelation_plot(data.series_daily_f107)
    plt.show()

    # model = ARIMA(y[:train_frac_idx], order=(2, 1, 2))
    # model_fit = model.fit()

    fh_max = 7  # predict up to this many days into the future
    test_idx_last = len(y) - (fh_max + 1)  # this will miss the last fh_max days of data for training purposes
    y_obs_all = []  # np.zeros((fh_max, test_idx_last - test_idx_first), np.float32)
    y_pred_all = []  # np.zeros((fh_max, test_idx_last - test_idx_first), np.float32)
    for fh in range(fh_max):  # forecast horizons (days)
        y_obs_all.append([])  # the observation record involved in each fh window
        y_pred_all.append([])  # the prediction in each fh window

    # walk forward validation across the data, increasing the size of the training set:
    n_tests = 0
    for t in random.sample(range(test_idx_first, test_idx_last), test_idx_last - test_idx_first):
        # train the model on the data up to this index:
        model = ARIMA(y[:t], order=(2, 1, 2))
        model_fit = model.fit()
        # output = model_fit.forecast() #1 day forecast
        for fh in range(fh_max):  # forecast horizons (days)
            y_pred = model_fit.predict(t, t + fh)
            # print(f"trained on {len(y[:t])} values, predictions of {t+1} to {t+fh+1}: y_pred = {y_pred}, observations: y_obs = {y_obs}")
            y_obs_all[fh].append(y[t:t + (fh + 1)])
            y_pred_all[fh].append(y_pred)
        n_tests += 1
        if n_tests == 20: break
    fherror = []
    for fh in range(fh_max):
        y_obs = np.array(y_obs_all[fh])
        y_pred = np.array(y_pred_all[fh])
        fherror.append(loss(torch.tensor(y_pred), torch.tensor(y_obs)).item())
        print(f'error with forecast horizon of {1 + fh} days = {fherror[-1]:.3f}')
    # error with forecast horizon of 1 days = 10.822
    # error with forecast horizon of 2 days = 17.347
    # error with forecast horizon of 3 days = 29.354
    # error with forecast horizon of 4 days = 52.861
    # error with forecast horizon of 5 days = 71.183
    # error with forecast horizon of 6 days = 96.630
    # error with forecast horizon of 7 days = 136.272


# hyper parameters:
size_batch = 20
seqlen = 20
learning_rate = 1e-4
n_hidden = 28
num_epochs = 25

lr_halves = 1

dataset = F107data(seqlen=seqlen, rollingtarget=True)  # 81, 74 without, with f30

# experiment_persistence(dataset);sys.exit()
# experiment_ARIMA(dataset);sys.exit()

# the first seqlen elements are not usable, because the don't have the full time history needed:
len_usable = len(dataset) - seqlen
n_train = int(0.8 * (len_usable))
n_test = len_usable - n_train
random_data_split = False
if random_data_split:
    # randomly split the train and test set:
    usable_set = torch.utils.data.Subset(dataset, range(seqlen, len(dataset)))
    train_set, test_set = torch.utils.data.random_split(usable_set, [n_train, n_test])
else:
    # non-randomly split the train and test set:
    train_set = torch.utils.data.Subset(dataset, range(seqlen, seqlen + n_train))
    test_set = torch.utils.data.Subset(dataset, range(seqlen + n_train, seqlen + n_train + n_test))

##ENSURE TRAINING DATASET IS SHUFFLED OR RANDOMLY SPLIT
train_loader = DataLoader(train_set, batch_size=size_batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=size_batch, shuffle=False)

X, y = next(iter(train_loader))
print("input like ", X.shape)  # batch size, sequence length, n features
print("target like ", y.shape)  # batch size, n_targets, target sequence length
n_features = X.shape[2]
n_targets = y.shape[2]

# define the model and learning rate:
model = mymodels.LSTM1_ts(n_features, n_targets, size_hidden=n_hidden)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# optimizer = torch.optim.LBFGS(model.parameters(), lr = 1e-3)#0.8)
# optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate)

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in data_loader:
            # X like [10, 27, 1]
            y_pred = model(X)  # dim: batch size, n_targets, seq len
            total_loss += loss_function(y_pred, y).item()
            # print(y_pred[0, -1, 0], y[0, -1, 0], y[0, -2, 0])
            # f107_fc = y_pred[:, -1, 0]  # last f10.7 value
            continue
            # total_loss += loss_function(f107_1dfc, f107_obs).item()

    avg_loss = total_loss / num_batches
    print(f" test loss: {avg_loss}")

    model.train()  # switch back into training mode
    return avg_loss


print("untrained test:")
test_model(test_loader, model, criterion)
print()

###################################
# main loop method:
# forward pass
# loss
# backward pass
# step parameters
# zero acculumated grads
# advance epoch and repeat...
###################################
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / size_batch)  # number of batches
# print(total_samples, n_iterations, len(train_loader))
print(f"{total_samples} samples into batches of size {size_batch} makes")
print(f"{len(train_loader)} training batches + {len(test_loader)} testing batches")
print()
loss_avg_prev = np.Inf
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):  # begins new iter() object
        # if (i+1)%100 ==0:
        #    print(f'epoch = {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

        def closure():
            optimizer.zero_grad()
            out = model(X)  # forward
            loss = criterion(out, y)
            # print("loss", loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)
        continue

    print(f"trained test at epoch {epoch}:")
    loss_avg = test_model(test_loader, model, criterion)
    if loss_avg > loss_avg_prev and lr_halves > 0:
        print("halving learning rate")
        lr_halves -= 1
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 2  # is this safe? what if 2 param groups point to the same 'lr'?
    loss_avg_prev = loss_avg

    # break #DELETE THIS
# plot prediction over test data:
if random_data_split:
    print("Cannot plot continuous forecast because training data is randomly ordered")
    sys.exit()

model.eval()

fh_max = 7
# Xaxis = np.empty((len(test_set)))
modelled = np.empty((fh_max, len(test_set)))
observed = np.empty((fh_max, len(test_set)))
epoch = np.empty((fh_max, len(test_set)))
epoch_selection = dataset.daily_epoch[seqlen + len(train_set):]
# epoch_datetime = np.array([datetime.datetime.fromtimestamp(x) for x in epoch_datetime])
with torch.no_grad():
    for it, (X, y) in enumerate(test_loader):
        X_input = X
        batchlen = X.shape[0]  # on the last iteration, batchlen <= size_batch
        idx_range = range(size_batch * it, size_batch * it + batchlen)

        observed[0, idx_range] = y[:, -1, 0]  # get the present values (fh=0)
        epoch[0, idx_range] = epoch_selection[idx_range]
        for fh in range(1, fh_max + 1):
            y_pred = model(X_input)  # dim: batch size, seq len, n_targets
            modelled[fh - 1, idx_range] = y_pred[:, -1, 0]  # last f10.7 values from each batch
            X_input = y_pred
# fill in the future obeservations:
for fh in range(1, fh_max):  # already filled in index 0
    observed[fh, :] = np.roll(observed[fh - 1, :], -1)
    observed[fh, -fh:] = np.nan
    epoch[fh, :] = np.roll(epoch[fh - 1, :], -1)
    epoch[fh, -fh:] = np.nan

# time = np.array([datetime.datetime.fromtimestamp(x) for x in dataset.daily_epoch[seqlen + len(train_set):]])
# f107_train = dataset.f107[seqlen + len(train_set):]
# print(f107_train[:10])
# print(observed[0,:10])
# print(epoch[0,:10])
# print([x.timestamp() for x in time[:10]])
# sys.exit()

# rescale:
modelled = (modelled * dataset.f107_s) + dataset.f107_m
observed = (observed * dataset.f107_s) + dataset.f107_m
# f107_train = (f107_train * dataset.f107_s) + dataset.f107_m

for fh in range(1, fh_max + 1):
    dt_1day = datetime.timedelta(days=1)
    fig = plt.figure(figsize=(24, 10))
    ax = plt.gca()
    fs = 20
    plt.title(f"Model {fh} Day Forecast", fontdict={'fontsize': fs})
    # plt.xlabel("day of testing data")
    plt.ylabel("F10.7 [sfu]", fontdict={'fontsize': fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    axis_t = [datetime.datetime.fromtimestamp(x) for x in epoch[fh - 1, :-fh]]
    plt.plot(axis_t, observed[fh - 1, :-fh], 'black', linewidth=0.5)
    # plt.plot(time, y_obs, 'blue', linewidth=0.5, alpha = 0.5)
    plt.plot(axis_t, modelled[fh - 1, :-fh], 'red', linewidth=0.5, alpha=0.5)

    fig.autofmt_xdate()

    # calculate persistence RMSE:
    index_range = range(1, len(epoch_selection) - 2 * fh)  # usable epoch array when doing a walk forward validation
    y_obs_all = []
    y_pred_all = []
    for idx in index_range:
        y_obs_all.append(observed[fh - 1, idx:idx + fh])
        y_pred_all.append(np.repeat(observed[fh - 1, idx - 1:idx], fh))  # copy previous day's value
    RMSE_pers = math.sqrt(MSE(np.array(y_obs_all), np.array(y_pred_all)))

    # calculate RMSE:
    y_obs_all = []
    y_pred_all = []
    for idx in index_range:
        y_obs_all.append(observed[fh - 1, idx:idx + fh])
        y_pred_all.append(modelled[fh - 1, idx:idx + fh])
    RMSE = math.sqrt(MSE(np.array(y_obs_all), np.array(y_pred_all)))
    print(f"fh = {fh}d: ", RMSE)

    # print hyperparameters
    hyperparams_score = f'batch size = {size_batch}\n' \
                        f'seq. len = {seqlen}\n' \
                        f'learning rate = {learning_rate:.3E}\n' \
                        f'no. hidden layers = {n_hidden}\n' \
                        f'no. epochs = {num_epochs}\n' \
                        f'RMSE_pers. = {RMSE_pers:.4f}\n' \
                        f'RMSE = {RMSE:.4f}'
    plt.text(0.01, 0.99, hyperparams_score, ha='left', va='top', transform=ax.transAxes, fontdict={'fontsize': fs})
    # plt.text(0.01, 0.99, f'RMSE = {RMS:.4f}', ha='left', va='top', transform = ax.transAxes, fontdict={'fontsize':fs})
    ax.grid(which='both')
    plt.xlim([datetime.date(2014, 1, 1), datetime.date(2015, 1, 1)])
    # plt.xlim([len(timeidx)-5,len(timeidx)+(1+fh)])
    plt.ylim([0, 300])
    plt.tight_layout()

    fname = f'predict_fh{fh}.png'
    plt.savefig(fname, dpi=300)
    print("output", fname)
    plt.close()
