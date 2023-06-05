import torch, torchvision
from torch import nn
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import math
from torch import nn, Tensor
import torch.nn.functional as F
target_mode_options = ["IMS", "DMS"]

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:
    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self,
        dropout: float = 0.1,
        seqlen_max: int = 5000,
        d_model: int = 512,
        batch_first: bool = False,
        device: str = "cpu"
    ):
        """
        Parameters:
            dropout: the dropout rate
            seqlen_max: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        self.device = device

        position = torch.arange(seqlen_max).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seqlen_max, d_model, device=self.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        #print(pe.size());sys.exit() # [1, seqlen_max, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size, seqlen_enc, dim_val]
        """
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

class tstransformer(nn.Module):
    def __init__(self,
                 n_features: int,
                 seqlen_enc: int,
                 seqlen_dec: int,
                 seqlen_out: int,
                 batch_first: bool,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 #dim_feedforward_encoder: int = 2048,
                 #dim_feedforward_decoder: int = 2048,
                 device: str = "cpu",
                 mode_target=target_mode_options[0]
                 ):
        """
        Args:
            n_features: int, number of input variables. 1 if univariate.
            seqlen_dec: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer of the decoder
            num_predicted_features: int, the number of features you want to predict.
        """

        super().__init__()
        self.n_features = n_features
        self.device = device
        self.mode_target = mode_target
        self.src_mask = generate_square_subsequent_mask(dim1=seqlen_out, dim2=seqlen_enc).to(self.device)
        self.tgt_mask = generate_square_subsequent_mask(dim1=seqlen_out, dim2=seqlen_out).to(self.device)

        dim_enc_feedforward = 1024 #2048 #default

        # encoder block
        #####################################################
        #linear input layer
        self.encoder_input_layer = nn.Linear(
            in_features=self.n_features,
            out_features=dim_val,
            device=self.device
        )
        #pe
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            seqlen_max=seqlen_enc,
            batch_first = batch_first,
            device=self.device
        )
        #
        #create encoder layers:
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = dim_val,
                                                         dim_feedforward=dim_enc_feedforward,
                                                         nhead=n_heads,
                                                         batch_first=batch_first,
                                                         dropout=dropout_encoder,
                                                         device=self.device)
        #stack the encoder layer n times in nn.TransformerDecoder:
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )
        #####################################################

        #decoder block
        #####################################################
        # linear input layer
        self.decoder_input_layer = nn.Linear(
            in_features=self.n_features,  # the number of features you want to predict. Usually just 1
            out_features=dim_val,
            device=self.device
        )

        #create decoder layers:
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, batch_first=batch_first, dropout=dropout_decoder,device=self.device)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
          )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=self.n_features,
            device=self.device
        )
        #####################################################

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """
        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        #change each sequence element to a vector
        src = self.encoder_input_layer(src)
        # [batch_size, seqlen_enc, n_features] -> [batch_size, seqlen_enc, dim_val] regardless of number of input features

        #positional encoding layer
        src = self.positional_encoding_layer(src)
        #[batch_size, seqlen_enc, dim_val] positionally encoded

        #pass through encoder layers
        # (self attention --> feed forward) Nx, the vector passed between ('sum' vector) has dimension dim_feedforward
        #self attention step 1)
        # multiple input (embedding) by 3 trained matricies -->
        # "query", a "key", and a "value" projection of each element in the sequence
        #
        #self attention step 2)
        # calculate attention scores for each element against each other element,
        # derived from the dot product of the first element's query with the other element's key (plus some other manipulation)
        # multiply the element's value by the attention score to get the 'sum' vector
        #
        #self attention multiple heads:
        # each encoder layer has n attention heads, which each produce their own 'sum' vector
        # these n vectors must be combined and fed into the same feed forward layer
        # to do this, they are concatenated then multiplied by a trained weights matrix
        # this weights matrix has row length dim_val so as to produce a set of vectors with the same dimensions as input embeddings
        #the feedforwad layer consists of 2 matrix multiplications using the dimension dim_feedforward,
        # but then also recovers the original input dimensions
        #see here for detailed dimenions: https://towardsdatascience.com/into-the-transformer-5ad892e0cee
        # masking is only needed in the encoder if input sequences are padded, but here, input sequences are of the same length
        src = self.encoder(src=src)  # src shape: [batch_size, seqlen_enc, dim_val]



        # Pass decoder input through decoder input layer)
        decoder_output = self.decoder_input_layer(tgt)  # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        # if src_mask is not None:
        # print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        # print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass through decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))
        return decoder_output

    # def get_src_trg(self,
    #                 sequence: torch.Tensor,
    #                 seqlen_enc: int,
    #                 target_seq_len: int) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    #     """
    #     Generate the src (encoder input), trg (decoder input) and trg_y (the target) sequences from a sequence.
    #     Args:
    #         sequence: tensor, a 1D tensor of length n where n = encoder input length + target sequence length
    #         seqlen_enc: int, the desired length of the input to the transformer encoder
    #         target_seq_len: int, the desired length of the target sequence (the
    #                         one against which the model output is compared)
    #     Return:
    #         src: tensor, 1D, used as input to the transformer model
    #         trg: tensor, 1D, used as input to the transformer model
    #         trg_y: tensor, 1D, the target sequence against which the model output is compared when computing loss.
    #     """
    #     #assert len(sequence) == seqlen_enc + target_seq_len, "Sequence length does not equal (input length + target length)"
    #
    #     # encoder input
    #     src = sequence[:seqlen_enc]
    #     # decoder input. As per the paper, it must have the same dimension as the
    #     # target sequence, and it must contain the last value of src, and all
    #     # values of trg_y except the last (i.e. it must be shifted right by 1)
    #     trg = sequence[seqlen_enc - 1:len(sequence) - 1]
    #     trg = trg[:, 0]
    #     #if len(trg.shape) == 1:
    #     #    trg = trg.unsqueeze(-1)
    #
    #     #assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"
    #
    #     # The target sequence against which the model output will be compared to compute loss
    #     trg_y = sequence[-target_seq_len:]
    #
    #     # We only want trg_y to consist of the target variable not any potential exogenous variables
    #     trg_y = trg_y[:, 0]
    #
    #     #assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"
    #     return src, trg, trg_y.squeeze(-1)  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]

    def test_model(self, data_loader, loss_function, scap):
        num_batches = len(data_loader)
        total_loss = 0

        self.eval()
        idx_target_priority = 0

        with torch.no_grad():
            for X, y in data_loader:
                X_scaled, y_scaled = scap(X, y)
                src = X_scaled
                trg_y = y_scaled
                trg = torch.cat((src[:, -1:], trg_y[:, :-1]), dim=1)

                # X like [10, 27, 1]
                out = self(src, trg, self.src_mask, self.tgt_mask)  # forward
                total_loss += loss_function(out, trg_y).item()
                # print(y_pred[0, -1, 0], y[0, -1, 0], y[0, -2, 0])
                # f107_fc = y_pred[:, -1, 0]  # last f10.7 value
                continue
                # total_loss += loss_function(f107_1dfc, f107_obs).item()

        avg_loss = total_loss / num_batches
        #print(f" test loss: {avg_loss}")

        self.train()  # switch back into training mode
        return avg_loss

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag. Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)




class LSTM1_ts(nn.Module): #modified for rolling output
    def __init__(self, seqlen, n_features, size_hidden, dropout, target_mode, num_layers = 1, device="cpu", seqlen_future = 1):
        super().__init__()
        self.seqlen = seqlen
        self.n_features = n_features
        self.size_hidden = size_hidden
        self.num_layers = num_layers
        self.device = device
        self.seqlen_future = seqlen_future

        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=self.size_hidden,
                            batch_first=True, #first dimension of input is batch size
                            num_layers=self.num_layers,
                            dropout=dropout,
                            device=self.device) #not used
        self.lstmcell1 = nn.LSTMCell(input_size=self.n_features,
                            hidden_size=self.size_hidden,
                            device=self.device)
        self.lstmcell2 = nn.LSTMCell(input_size=self.size_hidden,
                            hidden_size=self.size_hidden,
                            device=self.device)


        self.linearfinal = nn.Linear(in_features=self.size_hidden,
                                     out_features=self.n_features,
                                     device=self.device)

        if target_mode == "IMS":
            self.mask_of_input_in_output = torch.cat((torch.zeros(1, dtype=torch.bool), torch.ones(seqlen-1, dtype=torch.bool)),axis=0).to(device)
        elif target_mode == "DMS" or target_mode == "DMS_fh_max":
            self.mask_of_input_in_output = torch.zeros(seqlen, dtype=torch.bool).to(device)


    def forward(self, X):
        size_batch = X.shape[0]
        # the hidden and cell state tensors have batch size as the second dimension, even though batch_first=True
        h0 = torch.zeros(self.num_layers, size_batch, self.size_hidden, device=self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, size_batch, self.size_hidden, device=self.device).requires_grad_()

        outputs = torch.empty((size_batch, self.seqlen_future, self.n_features), device = self.device)

        for fh in range(self.seqlen_future):
            input = torch.cat((X[:, fh:, :], outputs[:, :fh, :]), dim=1)
            _, (hn, _) = self.lstm(input, (h0, c0))

            output = self.linearfinal(hn[self.num_layers-1])  # first dim of Hn is num_layers
            #output = torch.unsqueeze(output, 1) #COMMENTED 16/05/23
            outputs[:, fh, :] = output[:, :]

        return torch.cat((X[:, self.mask_of_input_in_output, :], outputs), dim=1)
    #     return self.arrange_output(X, outputs)
    #
    # def arrange_output_IMS(self, X, outputs):
    #     return torch.cat((X[:, 1:, :], outputs), dim=1)
    # def arrange_output_DMS(self, X, outputs):
    #     return outputs

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
    def test_model(self, data_loader, loss_function, scap):
        num_batches = len(data_loader)
        total_loss = 0

        self.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X_scaled, y_scaled = scap(X, y)

                out = self(X_scaled)  # dim: batch size, n_targets, seq len

                total_loss += loss_function(out, y_scaled).item()
                # print(y_pred[0, -1, 0], y[0, -1, 0], y[0, -2, 0])
                # f107_fc = y_pred[:, -1, 0]  # last f10.7 value
                continue
                # total_loss += loss_function(f107_1dfc, f107_obs).item()
        avg_loss = total_loss / num_batches
        #print(f" test loss: {avg_loss}")

        self.train()  # switch back into training mode
        return avg_loss