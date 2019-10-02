
import numpy as np
import torch.nn.functional as F

from torch import nn, torch

from LexTransformer.Modules import MultiHeadAttention, LexiconMultiHeadAttention, PointwiseFF

class TransformerEncoder(nn.Module):
    def __init__(self, num_head, d_x, d_k, d_linear, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.mh_attn = MultiHeadAttention(num_head=num_head, 
                                          d_x=d_x, d_k=d_k, 
                                          dropout=dropout)
        self.linear = PointwiseFF(d_x=d_x, 
                                  d_ff=d_linear, 
                                  dropout=dropout)
        
    def forward(self, X, pad_mask=None):
        output, attention = self.mh_attn(X, X, X, 
                                         pad_mask=pad_mask) 
        #output dim: batch * length_X * d_X
        #attention dim: batch * nh * l_q * l_k
        
        norm = X.ne(0.0).any(axis=2).type(torch.float).unsqueeze(-1)
        output *= norm
        
        
        output = self.linear(output) #batch * length * dx
        output *= norm
        
        return output, attention

    
class LexiconTransformerEncoder(nn.Module):
    def __init__(self, num_head, d_x, d_k, d_kl, d_linear, dropout=.1):
        super(LexiconTransformerEncoder, self).__init__()
        self.mh_attn = LexiconMultiHeadAttention(num_head=num_head, 
                                          d_x=d_x, d_k=d_k, d_kl=d_kl, 
                                          dropout=dropout)
        
        self.linear = PointwiseFF(d_x=d_x, 
                                  d_ff=d_linear, 
                                  dropout=dropout)
        
    def forward(self, X, z, pad_mask=None, pad_mask_l=None, context_mask=None):
        
        output, attention_context, attention_lexicon = self.mh_attn(X, X, X, z, z, 
                                         pad_mask=pad_mask,
                                         pad_mask_l=pad_mask_l,
                                         context_mask=context_mask) #output dim: batch * length_X * d_X

        #attention dim: batch * nh * l_q * l_k
        
        norm = X.ne(0.0).any(axis=2).type(torch.float).unsqueeze(-1)
        output *= norm
        
        
        output = self.linear(output) #batch * length * dx
        output *= norm
        
        return output, attention_context, attention_lexicon