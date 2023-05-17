import torch
import math
import numpy as np
import sys

class ScaleApplier:
    def __init__(self, data, dim_data_timeseq, dim_X_batch, dim_X_timeseq, dim_X_features):#, target_range_keep = None):
        self.dim_X_batch = dim_X_batch
        self.dim_X_timeseq = dim_X_timeseq
        self.dim_X_features = dim_X_features
        mean = torch.mean(data, dim=dim_data_timeseq)
        stddev = torch.std(data, dim=dim_data_timeseq)
        maxval = torch.max(data, dim=dim_data_timeseq).values
        minval = torch.min(data, dim=dim_data_timeseq).values

        # #store the indicies of the target to return for scaling operations
        # if target_range_keep is not None:
        #     target_range_keep = target_range_keep
        # else:
        #     target_range_keep = torch.arange(len(mean))
        # self.target_range_start = target_range_keep[0]
        # self.target_range_size = len(target_range_keep)

        #manipulate the above quantities to have the same dimension as X:
        applydims = {}
        applydims[dim_X_batch] = 1#size_X_batch
        applydims[dim_X_timeseq] = 1#size_X_timeseq
        applydims_keys = sorted(list(applydims.keys())) #must unsqueeze from lowest dim to highest
        for dim in applydims_keys:
            # insert a dimension at dim:
            mean = torch.unsqueeze(mean, dim)
            stddev = torch.unsqueeze(stddev, dim)
            maxval = torch.unsqueeze(maxval, dim)
            minval = torch.unsqueeze(minval, dim)

            newdims = list(mean.size())
            newdims[dim] = applydims[dim] #expand along the inserted dim to be the same size as sampleX

            mean = mean.expand(*newdims)
            stddev = stddev.expand(*newdims)
            maxval = maxval.expand(*newdims)
            minval = minval.expand(*newdims)

        self.mean = mean
        self.stddev = stddev
        self.mean_y = mean #mean.narrow(self.dim_X_features, self.target_range_start, self.target_range_size)
        self.stddev_y = stddev #stddev.narrow(self.dim_X_features, self.target_range_start, self.target_range_size)
        self.maxval = maxval
        self.minval = minval
        self.maxval_y = maxval #maxval.narrow(self.dim_X_features, self.target_range_start, self.target_range_size)
        self.minval_y = minval #minval.narrow(self.dim_X_features, self.target_range_start, self.target_range_size)

    def __call__(self, X, y):
        raise NotImplementedError

    def undo(self, X_scaled, y_scaled):
        raise NotImplementedError
class ScaleApplierStandard(ScaleApplier):
    def __init__(self, data, dim_data_timeseq, dim_X_batch = 0, dim_X_timeseq = 1, dim_X_features = 2):#, target_range_keep = None):
        super().__init__(data, dim_data_timeseq, dim_X_batch, dim_X_timeseq, dim_X_features)#, target_range_keep)

    def __call__(self, X, y):
        #size_X_batch_actual = X.shape[self.dim_X_batch] #may be smaller than the maximum batch size
        #mean = self.mean#.narrow(self.dim_X_batch, 0, size_X_batch_actual)
        #stddev = self.stddev#.narrow(self.dim_X_batch, 0, size_X_batch_actual)

        X_scaled = (X - self.mean)/self.stddev
        y_scaled = (y - self.mean_y)/self.stddev_y
        return X_scaled, y_scaled

    def undo(self, X_scaled, y_scaled):
        #size_X_batch_actual = X_scaled.shape[self.dim_X_batch]
        #mean = self.mean#.narrow(self.dim_X_batch, 0, size_X_batch_actual)
        #stddev = self.stddev#.narrow(self.dim_X_batch, 0, size_X_batch_actual)

        X = (X_scaled * self.stddev) + self.mean
        y = (y_scaled * self.stddev_y) + self.mean_y
        return X, y

class ScaleApplierNorm(ScaleApplier):
    def __init__(self, data, dim_data_timeseq, dim_X_batch = 0, dim_X_timeseq = 1, dim_X_features = 2):#, target_range_keep = None):
        super().__init__(data, dim_data_timeseq, dim_X_batch, dim_X_timeseq, dim_X_features)#, target_range_keep)

    def __call__(self, X, y):
        #size_X_batch_actual = X.shape[self.dim_X_batch] #may be smaller than the maximum batch size
        #maxval = self.maxval#.narrow(self.dim_X_batch, 0, size_X_batch_actual)
        #minval = self.minval#.narrow(self.dim_X_batch, 0, size_X_batch_actual)
        X_scaled = 2*(X - self.minval)/(self.maxval - self.minval) - 1
        y_scaled = 2*(y - self.minval_y)/(self.maxval_y - self.minval_y) - 1
        return X_scaled, y_scaled

    def undo(self, X_scaled, y_scaled):
        #size_X_batch_actual = X_scaled.shape[self.dim_X_batch]
        #maxval = self.maxval#.narrow(self.dim_X_batch, 0, size_X_batch_actual)
        #minval = self.minval#.narrow(self.dim_X_batch, 0, size_X_batch_actual)

        X = ((X_scaled + 1)/2)*(self.maxval - self.minval) + self.minval
        y = ((y_scaled + 1)/2)*(self.maxval_y - self.minval_y) + self.minval_y
        return X, y




# class ScaleApplier:
#     def __init__(self, data, dim_data_timeseq, size_X_batch, dim_X_batch, dim_X_timeseq, ):
#         self.dim_X_batch = dim_X_batch
#         self.dim_X_timeseq = dim_X_timeseq
#         mean = torch.mean(data, dim=dim_data_timeseq)
#         stddev = torch.std(data, dim=dim_data_timeseq)
#         maxval = torch.max(data, dim=dim_data_timeseq).values
#         minval = torch.min(data, dim=dim_data_timeseq).values
#
#         #manipulate the above quantities to have the same dimensions as X, y for a maximum batch size (with some degree of generality):
#         applydims = {}
#         applydims[dim_X_batch] = size_X_batch
#         applydims[dim_X_timeseq] = 1#size_X_timeseq
#         applydims_keys = sorted(list(applydims.keys())) #must unsqueeze from lowest dim to highest
#         for dim in applydims_keys:
#             # insert a dimension at dim:
#             mean = torch.unsqueeze(mean, dim)
#             stddev = torch.unsqueeze(stddev, dim)
#             maxval = torch.unsqueeze(maxval, dim)
#             minval = torch.unsqueeze(minval, dim)
#
#             newdims = list(mean.size())
#             newdims[dim] = applydims[dim] #expand along the inserted dim to be the same size as sampleX
#
#             mean = mean.expand(*newdims)
#             stddev = stddev.expand(*newdims)
#             maxval = maxval.expand(*newdims)
#             minval = minval.expand(*newdims)
#
#         self.mean = mean
#         self.stddev = stddev
#         self.maxval = maxval
#         self.minval = minval
#
#     def __call__(self, X, y):
#         raise NotImplementedError
#
#     def undo(self, X_scaled, y_scaled):
#         raise NotImplementedError
# class ScaleApplierStandard(ScaleApplier):
#     def __init__(self, data, dim_data_timeseq, size_X_batch, dim_X_batch = 0, dim_X_timeseq = 1):
#         super().__init__(data, dim_data_timeseq, size_X_batch, dim_X_batch = 0, dim_X_timeseq = 1)
#
#     def __call__(self, X, y):
#         size_X_batch_actual = X.shape[self.dim_X_batch] #may be smaller than the maximum batch size
#         mean = self.mean.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#         stddev = self.stddev.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#         X_scaled = (X - mean)/stddev
#         y_scaled = (y - mean)/stddev
#         return X_scaled, y_scaled
#
#     def undo(self, X_scaled, y_scaled):
#         size_X_batch_actual = X_scaled.shape[self.dim_X_batch]
#         mean = self.mean.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#         stddev = self.stddev.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#
#         X = (X_scaled *stddev) + mean
#         y = (y_scaled *stddev) + mean
#         return X, y
#
# class ScaleApplierNorm(ScaleApplier):
#     def __init__(self, data, dim_data_timeseq, size_X_batch, dim_X_batch = 0, dim_X_timeseq = 1):
#         super().__init__(data, dim_data_timeseq, size_X_batch, dim_X_batch = 0, dim_X_timeseq = 1)
#
#     def __call__(self, X, y):
#         size_X_batch_actual = X.shape[self.dim_X_batch] #may be smaller than the maximum batch size
#         maxval = self.maxval.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#         minval = self.minval.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#
#         X_scaled = 2*(X - minval)/(maxval - minval) - 1
#         y_scaled = 2*(y - minval)/(maxval - minval) - 1
#         return X_scaled, y_scaled
#
#     def undo(self, X_scaled, y_scaled):
#         size_X_batch_actual = X_scaled.shape[self.dim_X_batch]
#         maxval = self.maxval.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#         minval = self.minval.narrow(self.dim_X_batch, 0, size_X_batch_actual)
#
#         X = ((X_scaled + 1)/2)*(maxval - minval) + minval
#         y = ((y_scaled + 1)/2)*(maxval - minval) + minval
#         return X, y