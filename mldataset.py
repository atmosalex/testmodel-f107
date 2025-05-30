import torch
import numpy as np
import pandas as pd

class F107data:  # inherit the torch.utils.data.Dataset class
    def __init__(self, rawdata, target_mode, seqlen=5, seqlen_future = 1, device="cpu", features_use = None):
        self.target_mode = target_mode
        self.sl = seqlen
        # if target_mode == target_mode_options[0]:
        #     seqlen_future = 1
        self.sl_future = seqlen_future
        self.device = device

        #decide n_features:
        if features_use is None:
            self.features_use = rawdata.data.columns.values
        else:
            self.features_use = features_use
        self.epoch = rawdata.epoch
        df = pd.DataFrame(rawdata.data, columns=self.features_use)
        self.data = torch.tensor(df.values.astype(np.float32), device=self.device)
        self.column_names = df.columns.values


        #decide whether to shift input forward or get a sequence from the future for output
        if target_mode == "IMS":
            self.mask_of_input_in_output = torch.cat((torch.zeros(1, dtype=torch.bool), torch.ones(seqlen-1, dtype=torch.bool)),axis=0).to(device) #.repeat(len(self.features_use), 1).T
            #[0, 1, 1, 1, 1, ...]
            #self.arrange_output = self.arrange_output_IMS
            #self.target_range_keep = torch.arange(len(self.column_names))
            #self.n_targets = len(self.column_names)
        elif target_mode == "DMS" or target_mode == "DMS_fh_max":
            self.mask_of_input_in_output = torch.zeros(seqlen, dtype=torch.bool).to(device)
            # [0, 0, 0, 0, 0, ...]
            #self.arrange_output = self.arrange_output_DMS
            #self.target_range_keep = torch.arange(len(self.column_names))
            #self.n_targets = len(self.column_names)

        # elif target_mode == target_mode_options[2]: #"DMS_single_target"
        #     self.arrange_output = self.arrange_output_DMS_single_target
        #     self.target_range_keep = torch.tensor([idx_of_single_target])
        #     self.n_targets = 1
        #     print(f"idx_target_if_scalar set: this model will only predict {self.column_names[idx_of_single_target]} as a scalar")

    def __getitem__(self, idx):
        # INPUTS/FEATURES:
        if idx - self.sl >= 0:
            inputs = self.data[idx - self.sl: idx, :]
        else:
            # pad the time series with the value at index 0:
            inputs = torch.cat((self.data[0, :] * torch.ones((self.sl - idx, len(self.features_use)), device=self.device), self.data[:idx, :]))

        # TARGETS:
        if idx+self.sl_future <= len(self.data):
            target = self.data[idx:idx+self.sl_future, :]
        else:
            # pad the time series with the value at index -1:
            len_overshoot = idx+self.sl_future - len(self.data)
            target = torch.cat((self.data[idx:, :], self.data[len(self.data)-1, :] * torch.ones((len_overshoot, len(self.features_use)), device=self.device)))

        return inputs, torch.cat((inputs[self.mask_of_input_in_output, :], target), 0)
    #     return self.arrange_output(inputs, target)
    #
    # def arrange_output_IMS(self, i, o): #return vector target (sequence at times of input shifted forward by 1)
    #     return i, torch.cat((i[1:, :], o), 0)
    # def arrange_output_DMS(self, i, o): #return vector target (sequence at times of input shifted forward by 1)
    #     return i, o
    # # def arrange_output_DMS_single_target(self, i, o): #return scalar target (next target feature in sequence)
    # #     return i, o[:, self.target_range_keep]

    def __len__(self):
        return len(self.data)