import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
target_mode_options = ["shift_input_fwd", "future_sequence"]#, "future_sequence_single_target"]

class F107data:  # inherit the torch.utils.data.Dataset class
    def __init__(self, rawdata, target_mode, seqlen=5, seqlen_future = 1, device="cpu", idxrange_features_use = None):
        self.target_mode = target_mode
        self.sl = seqlen
        if target_mode == target_mode_options[0]:
            seqlen_future = 1
        self.sl_future = seqlen_future
        self.device = device

        self.epoch = rawdata.epoch
        self.data = torch.tensor(rawdata.data.values.astype(np.float32), device=self.device)
        self.column_names = rawdata.data.columns.values

        #decide n_features:
        if idxrange_features_use is None:
            self.idxrange_features_use = range(len(self.column_names))
        else:
            self.idxrange_features_use = idxrange_features_use

        #decide whether to shift input forward or get a sequence from the future for output
        if target_mode == target_mode_options[0]: #"shift_input_fwd"
            self.arrange_output = self.arrange_output_shift_input_fwd
            self.target_range_keep = torch.arange(len(self.column_names))
            self.n_targets = len(self.column_names)
        elif target_mode == target_mode_options[1]: #"future_sequence"
            self.arrange_output = self.arrange_output_future_sequence
            self.target_range_keep = torch.arange(len(self.column_names))
            self.n_targets = len(self.column_names)
        # elif target_mode == target_mode_options[2]: #"future_sequence_single_target"
        #     self.arrange_output = self.arrange_output_future_sequence_single_target
        #     self.target_range_keep = torch.tensor([idx_of_single_target])
        #     self.n_targets = 1
        #     print(f"idx_target_if_scalar set: this model will only predict {self.column_names[idx_of_single_target]} as a scalar")

    def __getitem__(self, idx):
        # INPUTS/FEATURES:
        if idx - self.sl >= 0:
            inputs = self.data[idx - self.sl: idx, self.idxrange_features_use]
        else:
            # pad the time series with the value at index 0:
            inputs = torch.cat((self.data[0, self.idxrange_features_use] * torch.ones((self.sl - idx, len(self.idxrange_features_use)), device=self.device), self.data[:idx, self.idxrange_features_use]))

        # TARGETS:
        if idx+self.sl_future <= len(self.data):
            target = self.data[idx:idx+self.sl_future, self.idxrange_features_use]
        else:
            # pad the time series with the value at index -1:
            len_overshoot = idx+self.sl_future - len(self.data)
            target = torch.cat((self.data[idx:, self.idxrange_features_use], self.data[len(self.data)-1, self.idxrange_features_use] * torch.ones((len_overshoot, len(self.idxrange_features_use)), device=self.device)))

        return self.arrange_output(inputs, target)

    def arrange_output_shift_input_fwd(self, i, o): #return vector target (sequence at times of input shifted forward by 1)
        return i, torch.cat((i[1:, :], o), 0)
    def arrange_output_future_sequence(self, i, o): #return vector target (sequence at times of input shifted forward by 1)
        return i, o
    # def arrange_output_future_sequence_single_target(self, i, o): #return scalar target (next target feature in sequence)
    #     return i, o[:, self.target_range_keep]

    def __len__(self):
        return len(self.data)