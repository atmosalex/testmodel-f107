import numpy as np
import pandas as pd
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plot
import torch
import random

def experiment_persistence(dataset, data, test_frac=0.2):
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
    #plot.autocorrelation_plot(data.series_daily_f107)


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