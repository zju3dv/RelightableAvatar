import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from lib.utils.net_utils import make_buffer


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=30, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = make_buffer(pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.shape[-2], :]  # B, L, D + B, L, D
        return self.dropout(x)


class MergingTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, max_len, in_dim, d_model=256, nhead=4, dim_feedforward=256, num_encoder_layers=4, dropout=0.0):
        super(MergingTransformer, self).__init__()
        self.d_model = d_model
        self.linear_mapping = nn.Linear(in_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.linear_decoder = nn.Conv1d(max_len, 1, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [batch size, sequence length, embed dim]
        # out: [batch size, embed dim]
        src = src * np.sqrt(self.d_model)  # B, L, D
        src = self.linear_mapping(src)
        src = self.positional_encoding(src)  # B, L, D
        out = self.transformer_encoder(src)  # B, L, D
        out = self.linear_decoder(out)  # B, 1, D
        return out  # B, 1, D


def generate_square_subsequent_mask(sz1, sz2, device='cuda'):
    return torch.triu(torch.full((sz1, sz2), float('-inf'), device=device), diagonal=1)


class SequenceTransformer(nn.Module):

    def __init__(self, in_dim, max_len=30, d_model=256, dropout=0.0):
        super(SequenceTransformer, self).__init__()
        self.d_model = d_model
        self.linear_mapping = nn.Linear(in_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model,
                                          dropout=dropout,
                                          batch_first=True,
                                        #   nhead=4,
                                        #   num_encoder_layers=4,
                                        #   num_decoder_layers=4,
                                        #   dim_feedforward=256,
                                          )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: B, L1, D
        # tgt: B, L2, D
        # out: B, L2, D

        B, L1, D = src.shape
        B, L2, D = tgt.shape
        mem_msk = generate_square_subsequent_mask(L2, L1, src.device)  # prevent the network from looking forward in time?
        tgt_msk = generate_square_subsequent_mask(L2, L2, src.device)

        src = self.linear_mapping(src)  # convert input to the required embedding dimension
        src = self.positional_encoding(src)  # add positional information to the input variables
        tgt = self.positional_encoding(tgt)  # add positional information to the target variables
        tgt = self.transformer(src, tgt,
                               memory_mask=mem_msk,
                               tgt_mask=tgt_msk,
                               )  # B, L2, D
        return tgt
