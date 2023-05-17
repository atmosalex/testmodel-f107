import copy
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
from sklearn.metrics import mean_squared_error as MSE
import models_nn as mymodels
#import models_transformer as mymodels
import loader
import mldataset
import plot
import scaler


dir_plots = "plotting"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
torch.manual_seed(99)

#set mode:
#
mode = mldataset.target_mode_options[0]
fh_max = 1
#
##

rawdata = loader.Solardata()
idxrange_features_use = torch.arange(1)

#print("warning: dataset has been scaled, with scaling factors influenced by testing dataset - this is bad practise")
print()
print("dataset is shape", rawdata.data.shape, "(time, features)")

# hyper parameters:
info_model = {}
info_model["num_epochs"] = 5; num_epochs = info_model["num_epochs"]
info_model["size_batch"] = 20; size_batch = info_model["size_batch"]
info_model["seqlen"] = 24; seqlen = info_model["seqlen"]
info_model["learning_rate_start"] = 1e-4; learning_rate = info_model["learning_rate_start"]
info_model["n_hidden"] = 28; n_hidden = info_model["n_hidden"]
info_model["scaling_method"] = 1; scaling_method = info_model["scaling_method"]
info_model["patience"] = 50; patience = info_model["patience"]
info_model["dropout"] = 0.5; dropout = info_model["dropout"]
info_model["seqlen_future"] = fh_max; seqlen_future = info_model["seqlen_future"]

#epoch interval to print training score:
print_interval = 1

#create pytorch dataset:
dataset = mldataset.F107data(rawdata, mode, seqlen=seqlen, seqlen_future = seqlen_future, idxrange_features_use = idxrange_features_use)

# the first seqlen elements are not usable, because they don't have the full time history:
len_usable = len(dataset) - seqlen
n_train = int(0.8 * (len_usable))
n_test = len_usable - n_train

# non-randomly split the train and test set:
train_indicies = range(seqlen, seqlen + n_train)
test_indicies = range(seqlen + n_train, seqlen + n_train + n_test)
train_set = torch.utils.data.Subset(dataset, train_indicies)
test_set = torch.utils.data.Subset(dataset, test_indicies)

#scaler:
ScaleAppliers = [scaler.ScaleApplierStandard, scaler.ScaleApplierNorm]
#scap = ScaleAppliers[scaling_method](dataset.data[train_indicies, :], dim_data_timeseq=0, size_X_batch=size_batch, dim_X_batch=0, dim_X_timeseq=1)
scap = ScaleAppliers[scaling_method](dataset.data[train_indicies, idxrange_features_use], dim_data_timeseq=0, dim_X_batch=0, dim_X_timeseq=1, dim_X_features=2)#, target_range_keep=dataset.target_range_keep)

#shuffle training data:
train_loader = DataLoader(train_set, batch_size=size_batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=size_batch, shuffle=False)

print("warning: test data is being used for validation - this is bad practise")
print()

X, y = next(iter(train_loader))
print("dataset iterations:")
print("input like ", X.shape)  # batch size, sequence length, n features
print("target like ", y.shape)  # batch size, n_targets, target sequence length
print()
n_features = len(idxrange_features_use)
n_targets = len(idxrange_features_use)

# define the model and learning rate:
target_names=[dataset.column_names[idxrange_features_use]]
print("targets:", target_names)
#LSTM:
model = mymodels.LSTM1_ts(n_features, size_hidden=n_hidden, dropout=dropout, num_layers=2, mode=mode, seqlen_future=seqlen_future)

# #transformer:
# dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
# n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
# n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
# n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
# model = mymodels.tstransformer(n_features=n_features,
#                                seqlen_enc=seqlen,
#                                seqlen_dec=seqlen_future, #ny
#                                seqlen_out=seqlen_future, #ny
#                                batch_first=True,
#                                dim_val=dim_val,
#                                n_decoder_layers=n_decoder_layers,
#                                n_encoder_layers=n_encoder_layers,
#                                n_heads=n_heads,
#                                device=device
#                                )


X_scaled, y_scaled = scap(X, y)
#sequence_batch = torch.cat((X, y), dim=1)
#src, trg, trg_y = model.get_src_trg(sequence_batch[0], seqlen, seqlen_future)


src = X_scaled
trg_y = y_scaled
trg = torch.cat((src[:,-1:], trg_y[:,:-1]), dim = 1)

# print()
# print(src.size())
# print(trg.size())
# print(trg_y.size())
# sys.exit()

#out = model(src, trg, src_mask, tgt_mask)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###################################
# main loop method:
# forward pass
# loss
# backward pass
# step parameters
# zero acculumated grads
# advance epoch and repeat...
###################################

n_iterations = math.ceil(len_usable / size_batch)  # number of batches
# print(total_samples, n_iterations, len(train_loader))
print(f"{len_usable} samples into batches of size {size_batch} makes")
print(f"{len(train_loader)} training batches + {len(test_loader)} testing batches")
print()
#print("untrained test:")
#loss_avg = mymodels.test_model(test_loader, model, criterion, scap)
loss_avg = np.inf
print(f"training losses (every {print_interval} epochs):")

epoch_improved = 0
loss_avg_best = loss_avg
model_best = copy.deepcopy(model)
for epoch in range(num_epochs):

    for i, (X, y) in enumerate(train_loader):
        X_scaled, y_scaled = scap(X, y)
        # n_features = 1:
        src = X_scaled
        trg_y = y_scaled
        trg = torch.cat((src[:, -1:], trg_y[:, :-1]), dim=1)

        def closure():
            optimizer.zero_grad()
            out = model(src)  # forward
            loss = criterion(out, trg_y)
            # print("loss", loss.item())
            loss.backward()
            return loss

        # def closure_transformer():
        #     optimizer.zero_grad()
        #     out = model(src, trg, model.src_mask, model.tgt_mask)  # forward
        #     loss = criterion(out, trg_y)
        #     # print("loss", loss.item())
        #     loss.backward()
        #     return loss

        # closure = closure_LSTM#_transformer
        optimizer.step(closure)
        continue

    loss_avg = mymodels.test_model(test_loader, model, criterion, scap)
    if loss_avg < loss_avg_best:
        epoch_improved = epoch
        loss_avg_best = loss_avg
        improvementstr = "*"
        model_best = copy.deepcopy(model)
    else:
        improvementstr = ""

    #print:
    if epoch % print_interval == 0:
        print("",f"#{str(epoch).zfill(4)}",loss_avg,"",improvementstr)

    if epoch - epoch_improved > patience:
        print("out of patience")
        break

    #break #TESTING - DELETE THIS LINE
model = model_best
print()



def perform_testing(model, info_model, test_loader):
    model.eval()
    size_batch = info_model["size_batch"]
    seqlen = info_model["seqlen"]
    learning_rate = info_model["learning_rate_start"]
    n_hidden = info_model["n_hidden"]
    scaling_method = info_model["scaling_method"]



    # get the epoch range corresponding to contiguous test data:
    time = test_set.dataset.epoch[test_set.indices]


    #loop through each feature in the dataset feature array:
    scoresummary = pd.DataFrame()
    scoresummary['fh'] = np.arange(fh_max) + 1

    #for target_idx, target_key in enumerate(target_names):
    for target_idx, target_key in enumerate(target_names):
        #dir_test = f"test_{target_key}"

        # use the model to forecast over the training data:
        forecasted = torch.empty((fh_max, len(test_set)), device=device)
        forecasted_persistence = torch.empty((fh_max, len(test_set)), device=device)
        observed = torch.empty((len(test_set)), device=device)  # stores values 1 day into the future

        with torch.no_grad():
            for it, (X, y) in enumerate(test_loader):
                batchlen = X.shape[0]  # on the last iteration, batchlen <= size_batch
                batchidxrange = torch.arange(size_batch * it, size_batch * it + batchlen, dtype=torch.long, device=device)

                forecasted_persistence[:fh_max, batchidxrange] = X[:, -1, target_idx]  # last values from each batch

                X_input, _ = scap(X, y)
                if mode == mymodels.target_mode_options[0]:
                    observed[batchidxrange] = y[:, -1, target_idx] #1 day ahead (current) value

                    for fh in range(fh_max):
                        y_pred = model(X_input)  # dim: batch size, seq len, n_targets
                        _, y = scap.undo(X_input, y_pred)

                        forecasted[fh, batchidxrange] = y[:, -1, target_idx]  # last values from each batch
                        X_input = torch.cat((X_input[:, 1:, :], y_pred[:, -1:, :]), dim=1)
                        #forecasted_persistence[:fh, batchidxrange] = X[:, -1, target_idx]  # last values from each batch
                elif mode == mymodels.target_mode_options[1]:
                    observed[batchidxrange] = y[:, 0, target_idx] #1 day ahead (current) value

                    y_pred = model(X_input)
                    _, y = scap.undo(X_input, y_pred)

                    forecasted[:fh_max, batchidxrange] = y[:, :fh_max, target_idx].T  # first values from each batch
                    #X_input = torch.cat((X_input[:, 1:, :], y_pred[:, 0:1, :]), dim=1)
                else:
                    print("error: mode not implemented for testing")
                    sys.exit()


        # convert torch results to numpy on CPU so we can plot:
        forecasted = forecasted.detach().cpu().numpy()
        forecasted_persistence = forecasted_persistence.detach().cpu().numpy()
        observed = observed.detach().cpu().numpy()

        observed = observed

        # evaluate scores:
        fh_RMSE = []
        fh_RMSE_persistence = []
        for fh in range(fh_max):
            obs = observed[fh:-1]
            fc = forecasted[fh, :-1 - fh]
            fc_p = forecasted_persistence[fh, :-1 - fh]
            RMSE = math.sqrt(np.mean(np.power(obs - fc, 2)))
            # RMSE = math.sqrt(criterion(observed[fh:-1], forecasted[fh, :-1 - fh]))
            RMSE_persistence = math.sqrt(np.mean(np.power(obs - fc_p, 2)))
            # RMSE_persistence = math.sqrt(criterion(observed[fh:-1], forecasted_persistence[fh, :-1 - fh]))
            fh_RMSE.append(RMSE)
            fh_RMSE_persistence.append(RMSE_persistence)
        fh_RMSE = np.array(fh_RMSE)
        fh_RMSE_persistence = np.array(fh_RMSE_persistence)

        #add the score ratio to our scoresummary table:
        scoresummary[target_key] = fh_RMSE/fh_RMSE_persistence

        # add to dictionary of model parameters (for plotting):
        info_model['target_key'] = target_key
        info_model['fh_RMSE'] = fh_RMSE
        info_model['fh_RMSE_persistence'] = fh_RMSE_persistence

        # if logy:
        #     # convert back to non-log space:
        #     forecasted = np.power(10, forecasted)
        #     forecasted_persistence = np.power(10, forecasted_persistence)
        #     observed = np.power(10, observed)

        plot.plot_forecast(dir_plots, time, observed, forecasted, forecasted_persistence, info_model,
                           plot_logy = False, subdir = target_key)

    scoresummary.set_index('fh')

    return scoresummary


scoresummary = perform_testing(model, info_model, test_loader)
print(f'RMSE:RMSE_p')
print(scoresummary.to_string(index=False))
