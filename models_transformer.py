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
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = False
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seqlen_enc, dim_val] or 
               [seqlen_enc, batch_size, dim_val]
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
                 device: str = "cpu"
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
        self.src_mask = generate_square_subsequent_mask(dim1=seqlen_out, dim2=seqlen_enc)
        self.tgt_mask = generate_square_subsequent_mask(dim1=seqlen_out, dim2=seqlen_out)

        # encoder block
        #####################################################
        #linear input layer
        self.encoder_input_layer = nn.Linear(
            in_features=self.n_features,
            out_features=dim_val
        )
        #pe
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=seqlen_enc,
            batch_first = batch_first
        )
        #
        #create encoder layers:
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = dim_val, nhead=n_heads, batch_first=batch_first,dropout=dropout_encoder)
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
            in_features=1,  # the number of features you want to predict. Usually just 1
            out_features=dim_val
        )

        #create decoder layers:
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, batch_first=batch_first, dropout=dropout_decoder)
        self.decoder = nn.TransformerDecoder(
          decoder_layer=decoder_layer,
          num_layers=n_decoder_layers,
          norm=None
          )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=self.n_features
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

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(src=src)  # src shape: [batch_size, seqlen_enc, dim_val]
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

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

def test_model(data_loader, model, loss_function, scap):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    idx_target_priority = 0

    with torch.no_grad():
        for X, y in data_loader:
            X_scaled, y_scaled = scap(X, y)
            src = X_scaled
            trg_y = y_scaled
            trg = torch.cat((src[:, -1:], trg_y[:, :-1]), dim=1)

            # X like [10, 27, 1]
            out = model(src, trg, model.src_mask, model.tgt_mask)  # forward
            total_loss += loss_function(out, trg_y).item()
            # print(y_pred[0, -1, 0], y[0, -1, 0], y[0, -2, 0])
            # f107_fc = y_pred[:, -1, 0]  # last f10.7 value
            continue
            # total_loss += loss_function(f107_1dfc, f107_obs).item()

    avg_loss = total_loss / num_batches
    #print(f" test loss: {avg_loss}")

    model.train()  # switch back into training mode
    return avg_loss

