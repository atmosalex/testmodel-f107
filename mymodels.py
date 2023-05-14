import torch, torchvision
from torch import nn
import sys
import numpy as np
class LSTM1_ts(nn.Module): #modified for rolling output
    def __init__(self, n_features, n_targets=1, size_hidden=16):
        super().__init__()
        self.n_features = n_features
        self.size_hidden = size_hidden
        self.num_layers = 1
        self.n_targets = n_targets

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.size_hidden,
                            batch_first=True, #first dimension of input is batch size
                            num_layers=self.num_layers) #not used
        self.lstmcell1 = nn.LSTMCell(input_size=self.n_features,
                            hidden_size=self.size_hidden)
        self.lstmcell2 = nn.LSTMCell(input_size=self.size_hidden,
                            hidden_size=self.size_hidden)

        self.relu1 = nn.ReLU()
        self.lrelu1 = nn.LeakyReLU()

        self.linearinitial = nn.Linear(in_features=self.n_targets, out_features=self.size_hidden)
        self.linearfinal = nn.Linear(in_features=self.size_hidden, out_features=self.n_targets)

    def forward(self, X):
        return self.forward_lstm1(X)

    def forward_lin(self, X):
        out = self.linearinitial(X)
        out = self.lrelu1(out)
        output = self.linearfinal(out)
        return output

    def forward_lstm1(self, X):
        #print(X.shape) #batch, seqlen, features, i.e. correct input dim for LSTM when batch_first=True
        size_batch = X.shape[0]

        #the hidden and cell state tensors have batch size as the second dimension
        h_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        c_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        h_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        c_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        outputs = []
        for input_t in X.split(1, dim=1):
            #lstmcell input has dims: batch size, n_features
            h_t, c_t = self.lstmcell1(input_t[:,0,:], (h_t, c_t))
            output = self.linearfinal(h_t)
            #h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
            #output = self.linear(h_t2)

            output = torch.unsqueeze(output, 1)
            outputs.append(output)
        #outputs = torch.reshape(outputs, (size_batch, len(outputs), self.n_features))
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward_lstm2(self, X):
        # print(X.shape) #batch, seqlen, features, i.e. correct input dim for LSTM when batch_first=True
        size_batch = X.shape[0]

        # the hidden and cell state tensors have batch size as the second dimension
        h_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        c_t = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        h_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        c_t2 = torch.zeros(size_batch, self.size_hidden).requires_grad_()
        outputs = []
        for input_t in X.split(1, dim=1):
            # lstmcell input has dims: batch size, n_features
            h_t, c_t = self.lstmcell1(input_t[:, 0, :], (h_t, c_t))
            h_t2, c_t2 = self.lstmcell2(h_t, (h_t2, c_t2))
            output = self.linearfinal(h_t2)

            output = torch.unsqueeze(output, 1)
            outputs.append(output)
        # outputs = torch.reshape(outputs, (size_batch, len(outputs), self.n_features))
        outputs = torch.cat(outputs, dim=1)
        return outputs



        # for i in range(future):
        #     #same thing - keep going using the previous output
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs.append(output)


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Constructor
    def __init__(
            self,
            num_tokens,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)