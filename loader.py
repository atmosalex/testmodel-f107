import os
import h5py
import numpy as np
import datetime
import math
import sys
import pandas as pd
import copy

class Solardata:
    def __init__(self, preprocessed_h5 = 'dataset.h5', forceload = False):
        if forceload or not os.path.exists(preprocessed_h5): #write the data to a .h5 file for faster loading
            # load the F10.7 flux data from https://lasp.colorado.edu/lisird/data/penticton_radio_flux/:
            date_parser = lambda x: datetime.datetime.strptime(x, '%Y%m%d_%H%M%S.%f')
            raw_f107 = pd.read_csv('penticton_radio_flux.csv', parse_dates=['time (yyyyMMdd_HHmmss.SSS)'], date_parser=date_parser)
            raw_f107.drop(['observed_flux (solar flux unit (SFU))'], axis = 1, inplace=True)
            raw_f107.rename(columns={'time (yyyyMMdd_HHmmss.SSS)':'Date'}, inplace=True)
            raw_f107.set_index('Date', inplace=True)
            raw_f107.dropna(inplace=True)

            # load the 30cm flux data from http://solar.nro.nao.ac.jp/norp/html/daily_flux.html:
            date_parser = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
            raw_f30 = pd.read_csv('TYKW-NoRP_dailyflux.txt', skiprows=1, parse_dates=['Date'], date_parser=date_parser)
            raw_f30.drop(['1 GHz', '2 GHz', '3.75 GHz'], axis = 1, inplace= True)
            raw_f30.set_index('Date', inplace=True)
            raw_f30.dropna(inplace=True)

            epoch0 = max(raw_f30.index[0], raw_f107.index[0])
            epoch1 = min(raw_f30.index[-1], raw_f107.index[-1])
            dr = pd.date_range(start=epoch0, end=epoch1, freq='1D')
            ts = dr.values.astype(int)/1e9

            df = pd.DataFrame({"epoch":ts})
            df['f107'] = np.interp(ts, raw_f107.index.values.astype(int)/1e9, raw_f107.values[:,0])
            df['f30'] = np.interp(ts, raw_f30.index.values.astype(int)/1e9, raw_f30.values[:,0])

            df.to_hdf(preprocessed_h5, key='df', mode='w')

        df = pd.read_hdf(preprocessed_h5)
        dt = np.array([datetime.timedelta(seconds=t) for t in df['epoch'].values])
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + dt
        self.epoch = epoch
        self.data = df.drop(['epoch'], axis=1)

        #scaling factors:
        self.data_m = self.data.values.mean(keepdims=True, axis = 0)
        self.data_s = self.data.values.std(keepdims=True, axis = 0)
        self.scaled = False
    def scale_inplace(self):
        self.scaled = True
        self.data -= self.data_m
        self.data /= self.data_s
    def unscale_inplace(self):
        self.data *= self.data_s
        self.data += self.data_m
        self.scaled = False
    def get_unscaled(self):
        data = copy.deepcopy(self.data)
        if self.scaled:
            data *= self.data_s
            data += self.data_m
        return data



