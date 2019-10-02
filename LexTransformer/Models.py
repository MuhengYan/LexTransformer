import torch

import torch.nn as nn
import numpy as np

from LexTransformer.Modules import Embed, PosEmbed
from LexTransformer.Encoders import TransformerEncoder, LexiconTransformerEncoder

class Encoder(nn.Module):
    def __init__(self, 
               length, emb_dim, embeddings, 
               num_layer, num_head, d_k, d_linear,
               d_kl = None, alpha=0.5, dropout=.1):
        super(Encoder, self).__init__()
        self.embed = Embed(length=length, emb_dim=emb_dim, 
                           embeddings=embeddings, 
                           trainable=False, dropout=dropout)
        
        self.pos_embed = PosEmbed(length=length, 
                                  emb_dim=emb_dim)
        
        if d_kl is None:
            self.include_lex = False
            print('Initializing Plain Transformer!')
            self.transformers = nn.ModuleList([TransformerEncoder(num_head=num_head, 
                                              d_x=emb_dim, d_k=d_k, 
                                              d_linear=d_linear, 
                                              dropout=dropout) for i in range(num_layer)])
        else:
            self.include_lex = True
            print('Initializing LexTransformer!')
            self.transformers = nn.ModuleList([LexiconTransformerEncoder(num_head=num_head, 
                                              d_x=emb_dim, d_k=d_k, 
                                              d_linear=d_linear,
                                              dropout=dropout) for i in range(num_layer)])
            

            
    def forward(self, X, z=None, context_mask=None):

        embedded = self.embed(X) + self.pos_embed(X, 'seq')
        
        pad_mask = self.masking(X, X)
        
        if z is not None:
            pad_mask_l = self.masking(X, z)
#             embedded_z = self.embed(z) + pos_emb(z, 'set')
            embedded_z = self.embed(z)

        
        encoded = embedded
        
        if self.include_lex:
            for _transformer in self.transformers:
                encoded, attn_con, attn_lex = _transformer(encoded, embedded_z, 
                                                      pad_mask=pad_mask, 
                                                      pad_mask_l=pad_mask_l, 
                                                      context_mask=context_mask)
        else:
            for _transformer in self.transformers:
                encoded, attn_con = _transformer(encoded, pad_mask=pad_mask)
            attn_lex = None
        
        return encoded, attn_con, attn_lex
        
    @staticmethod
    
    
    def masking(query_seq, key_seq):
        l_q = query_seq.shape[1]
        l_k = key_seq.shape[1]
        mask_q = query_seq.eq(0).unsqueeze(-1).expand(-1, -1, l_k)
        mask_k = key_seq.eq(0).unsqueeze(1).expand(-1, l_q, -1)
        mask = mask_q.masked_fill(mask_k, True)
        return mask
    
    
class DenseLayers(nn.Module):
    #task specific dense layer
    def __init__(self, dim, n_logits):
        
        super(DenseLayers, self).__init__()
        self.dense = nn.Linear(in_features=dim,
                                out_features=n_logits)
        
    
    def forward(self, X):
        return self.dense(X)
    
class LexiconTransformerClassifier(nn.Module):
    #full model
    def __init__(self, length, emb_dim, embeddings, 
                 n_transformer, num_head, d_k, 
                 d_linear, d_kl=None, alpha=0.5, 
                 n_logits=2,
                 dropout=.1):
        super(LexiconTransformerClassifier, self).__init__()
        
        self.length = length
        self.emb_dim = emb_dim
        
        self.encoder = Encoder(length=length, 
                               emb_dim=emb_dim, 
                               embeddings=embeddings, 
                               num_layer=n_transformer, 
                               num_head=num_head, 
                               d_k=d_k, 
                               d_linear=d_linear,
                               d_kl = d_kl, 
                               alpha=alpha, 
                               dropout=dropout)

        self.dense_layer = DenseLayers(dim=length*emb_dim, n_logits=n_logits)
        

    def forward(self, X, z=None, context_mask=None):
        encoded, attn_con, attn_lex = self.encoder(X=X, z=z, 
                                                   context_mask=context_mask)
        
        encoded = encoded.view(-1, self.length * self.emb_dim) #batch * (length*emb_dim)
        
        logits = self.dense_layer(encoded)
        
        return logits, attn_con, attn_lex
        
        