import torch, torchvision
from torch import nn
import sys
import numpy as np
import math

target_mode_options = ["shift_input_fwd", "future_sequence", "future_sequence_single_target"]

class LSTM1_ts(nn.Module): #modified for rolling output
    def __init__(self, n_features, size_hidden, dropout, num_layers = 1, device="cpu", mode=target_mode_options[0], seqlen_future = 1):
        super().__init__()
        self.n_features = n_features
        self.size_hidden = size_hidden
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.size_hidden,
                            batch_first=True, #first dimension of input is batch size
                            num_layers=self.num_layers, dropout=dropout) #not used
        self.lstmcell1 = nn.LSTMCell(input_size=self.n_features,
                            hidden_size=self.size_hidden)
        self.lstmcell2 = nn.LSTMCell(input_size=self.size_hidden,
                            hidden_size=self.size_hidden)


        self.linearfinal = nn.Linear(in_features=self.size_hidden, out_features=self.n_features)

        if mode == target_mode_options[0]:
            self.arrange_output = self.arrange_output_shift_input_fwd
            self.seqlen_future = 1
        elif mode == target_mode_options[1]:
            self.arrange_output = self.arrange_output_future_sequence
            self.seqlen_future = seqlen_future
        else:
            print(f"error: mode {mode} not supported by model forward()")
            sys.exit()

    def forward(self, X):
        size_batch = X.shape[0]
        # the hidden and cell state tensors have batch size as the second dimension, even though batch_first=True
        h0 = torch.zeros(self.num_layers, size_batch, self.size_hidden, device=self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, size_batch, self.size_hidden, device=self.device).requires_grad_()

        outputs = torch.empty((size_batch, self.seqlen_future, self.n_features))

        for fh in range(self.seqlen_future):
            input = torch.cat((X[:, fh:, :], outputs[:, :fh, :]), dim=1)
            _, (hn, _) = self.lstm(input, (h0, c0))

            output = self.linearfinal(hn[self.num_layers-1])  # first dim of Hn is num_layers
            #output = torch.unsqueeze(output, 1) #COMMENTED 16/05/23
            outputs[:, fh, :] = output[:, :]

        return self.arrange_output(X, outputs)

    def arrange_output_shift_input_fwd(self, X, outputs):
        return torch.cat((X[:, 1:, :], outputs), dim=1)
    def arrange_output_future_sequence(self, X, outputs):
        return outputs

    # def forward_lin(self, X):
    #     out = self.linearinitial(X)
    #     output = self.linearfinal(out)
    #     return output
    #
    # def forward_lstm1(self, X):
    #     #print(X.shape) #batch, seqlen, features, i.e. correct input dim for LSTM when batch_first=True
    #     size_batch = X.shape[0]
    #
    #     #the hidden and cell state tensors have batch size as the second dimension
    #     h_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     c_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     h_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     c_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     outputs = []
    #     for input_t in X.split(1, dim=1):
    #         #lstmcell input has dims: batch size, n_features
    #         h_t, c_t = self.lstmcell1(input_t[:,0,:], (h_t, c_t))
    #         output = self.linearfinal(h_t)
    #         #h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
    #         #output = self.linear(h_t2)
    #
    #         output = torch.unsqueeze(output, 1)
    #         outputs.append(output)
    #     #outputs = torch.reshape(outputs, (size_batch, len(outputs), self.n_features))
    #     outputs = torch.cat(outputs, dim=1)
    #     return outputs
    #
    # def forward(self, X):
    #     # print(X.shape) #batch, seqlen, features, i.e. correct input dim for LSTM when batch_first=True
    #     size_batch = X.shape[0]
    #
    #     # the hidden and cell state tensors have batch size as the second dimension
    #     h_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     c_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     h_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     c_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
    #     outputs = []
    #     for input_t in X.split(1, dim=1):
    #         # lstmcell input has dims: batch size, n_features
    #         #h_t = h_t2 #SOMETHING IS WRONG HERE - THESE TWO LINES DON'T MAKE MUCH DIFFERENCE TO SCORE, ACTUALLY THEY IMPROVED SCORE
    #         #c_t = c_t2
    #         h_t, c_t = self.lstmcell1(input_t[:, 0, :], (h_t, c_t))
    #         h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
    #         output = self.linearfinal(h_t2)
    #         #I think this is just training a cell to predict the next value based on the one prior
    #
    #         output = torch.unsqueeze(output, 1)
    #         outputs.append(output)
    #     # outputs = torch.reshape(outputs, (size_batch, len(outputs), self.n_features))
    #     outputs = torch.cat(outputs, dim=1)
    #     return outputs
    #     #0.0012318231042180011
def test_model(data_loader, model, loss_function, scap):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in data_loader:
            X_scaled, y_scaled = scap(X, y)

            out = model(X_scaled)  # dim: batch size, n_targets, seq len

            total_loss += loss_function(out, y_scaled).item()
            # print(y_pred[0, -1, 0], y[0, -1, 0], y[0, -2, 0])
            # f107_fc = y_pred[:, -1, 0]  # last f10.7 value
            continue
            # total_loss += loss_function(f107_1dfc, f107_obs).item()
    avg_loss = total_loss / num_batches
    #print(f" test loss: {avg_loss}")

    model.train()  # switch back into training mode
    return avg_loss