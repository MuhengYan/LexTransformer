#basic components: Embedding Layer, Scaled Dot-Product Attention, Dense Layer

import numpy as np
import torch.nn.functional as F
from torch import nn, torch


class Embed(nn.Module):
    def __init__(self, length, emb_dim, 
                 embeddings=None, trainable=False, dropout=.1):
        super(Embed, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=length, 
                                      embedding_dim=emb_dim, 
                                      padding_idx=0)
        
        if embeddings is not None:
            print("Loading pre-trained embeddings!")
            self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings),
                                                 requires_grad=trainable)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, X):
        embedded = self.embedding(X)
                
        embedded = self.dropout(embedded)
            
        return embedded
    
    
class PosEmbed(nn.Module):
    def __init__(self, length, emb_dim):
        super(PosEmbed, self).__init__()
        
        self.length = length
        self.emb_dim = emb_dim
        
        pos_weight = self.position_encoding_init(n_position=length,
                                                emb_dim=emb_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(pos_weight, freeze=True)
        
        
    def get_pos(self, word_sequences, mode='seq'):
        batch = []
        for word_seq in word_sequences:
            start_idx = 1.0
            word_pos = []
            for pos in word_seq:
                if mode == 'seq':
                    if int(pos) == 0:
                        word_pos.append(0.0)
                    else:
                        word_pos.append(start_idx)
                        start_idx += 1.0
                elif mode == 'set':
                    word_pos.append(0.0)
                else:
                    raise ValueError('Unrecognized position encoding mode! Should be chosen from "seq" or "set"! ')

                
            batch.append(torch.from_numpy(np.array(word_pos)).type(torch.LongTensor))
        batch = torch.cat(batch).view(-1, self.length)        
        return batch

    
    def forward(self, X, mode='seq'):
        X = self.get_pos(X, mode=mode)
        pos_embeded = self.pos_embedding(X)
        return pos_embeded
        
        
        
        
    @staticmethod
    def position_encoding_init(n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''
        # keep dim 0 for padding token position encoding zero vector
        n_position += 1
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)
    

    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        
        super(ScaledDotProductAttention, self).__init__()
        self.reg = np.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2) #input tensor dim: (batch, seq_length, seq_length)
        
    
    def forward(self, q, k, v, pad_mask=None, context_mask=None):
        
        attention = torch.bmm(q, k.transpose(1, 2)) #dim of q and k: (batch, seq_length, d_k * n_head)
        attention /= self.reg
        
        if pad_mask is not None:
            attention = attention.masked_fill(pad_mask, -1e9) #see Attention is all you need 3.2.3
            
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        
        if context_mask is not None: #context masking
            attention *= context_mask
        
        output = torch.bmm(attention, v)
        
        return output, attention


    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_x, d_k, dropout=.1):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_k = d_k
        
        self.wq = nn.Linear(d_x, num_head * d_k)
        self.wk = nn.Linear(d_x, num_head * d_k)
        self.wv = nn.Linear(d_x, num_head * d_k)    
        
        #initialization problems?
        
        self.sdp_attn = ScaledDotProductAttention(d_k=d_k, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_x)
        
        self.wo = nn.Linear(num_head * d_k, d_x)
#         nn.init.xavier_normal_(self.wo.weight)
        
    
    def forward(self, q, k, v, pad_mask=None):
        X = q #batch * length_q * d_x
        length_q = q.shape[1]
        assert v.shape[1] == k.shape[1]
        length_k = k.shape[1]
        
        q = self.wq(q).view(-1, length_q, self.num_head, self.d_k) #batch * length * num_head * d_k
        k = self.wk(k).view(-1, length_k, self.num_head, self.d_k)
        v = self.wv(v).view(-1, length_k, self.num_head, self.d_k) 
        
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, length_q, self.d_k) # (batch * num_head) * length * d_k
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, length_k, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, length_k, self.d_k)
        
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(self.num_head, 1, 1) # batch * length_q * length_k -> (batch * num_head) * l_q * l_k

        output, attention = self.sdp_attn(q, k, v, 
                                          pad_mask=pad_mask) 
        #output: (batch*nh) * length_q * d_k
        #attention: (batch*nh) * length_q * length_k
        
        output = output.view(-1, self.num_head, length_q, self.d_k) #batch * nh * l_q * d_k
        output = output.permute(0, 2, 1, 3).contiguous().view(-1, length_q, self.num_head * self.d_k) #batch * l_q * (nh * d_k)
        
        output = self.norm(self.dropout(self.wo(output)) + X) #batch * l_q * d_x
        
        attention = attention.view(-1, self.num_head, length_q, length_k) #batch * nh * l_q * l_k
        
        return output, attention
    
    
    
class LexiconMultiHeadAttention(nn.Module):
    
    def __init__(self, num_head, d_x, d_k, d_kl, dropout=.1):
        
        super(LexiconMultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_kl = d_kl
        
        self.wq = nn.Linear(d_x, num_head * d_k)
        self.wk = nn.Linear(d_x, num_head * d_k)
        self.wv = nn.Linear(d_x, num_head * d_k)   
        self.wkl = nn.Linear(d_x, num_head * d_kl)
        self.wvl = nn.Linear(d_x, num_head * d_kl)  
        
        #initialization problems?
        
        self.sdp_attn_context = ScaledDotProductAttention(d_k=d_k, dropout=dropout)
        self.sdp_attn_lex = ScaledDotProductAttention(d_k=d_kl, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_x)
        
        self.wo = nn.Linear(num_head * d_k, d_x)
#         nn.init.xavier_normal_(self.wo.weight)
        
    
    def forward(self, q, k, v, kl, vl, 
                pad_mask=None, pad_mask_l=None, 
                context_mask=None, alpha=0.5):
        
        X = q #batch * length_q * d_x
        length_q = q.shape[1]
        
        assert v.shape[1] == k.shape[1]
        length_k = k.shape[1]
        assert vl.shape[1] == kl.shape[1]
        length_kl = kl.shape[1]
        
        q = self.wq(q).view(-1, length_q, self.num_head, self.d_k) #batch * length * num_head * d_k
        
        k = self.wk(k).view(-1, length_k, self.num_head, self.d_k)
        v = self.wv(v).view(-1, length_k, self.num_head, self.d_k) 
        
        kl = self.wkl(kl).view(-1, length_kl, self.num_head, self.d_kl)
        vl = self.wvl(vl).view(-1, length_kl, self.num_head, self.d_kl) 
        
        
        
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, length_q, self.d_k) # (batch * num_head) * length * d_k
        
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, length_k, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, length_k, self.d_k)
        
        kl = kl.permute(0, 2, 1, 3).contiguous().view(-1, length_kl, self.d_kl)
        vl = vl.permute(0, 2, 1, 3).contiguous().view(-1, length_kl, self.d_kl)
        
        
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(self.num_head, 1, 1) # batch * length_q * length_k -> (batch * num_head) * l_q * l_k
            
        if pad_mask_l is not None:
            pad_mask_l = pad_mask_l.repeat(self.num_head, 1, 1)
            
        if context_mask is not None:
            value_mask = value_mask.repeat(self.num_head, 1, 1)
            
        output_context, attention_context = self.sdp_attn_context(q, k, v, 
                                                                  pad_mask=pad_mask)
        
        output_lexicon, attention_lexicon = self.sdp_attn_lex(q, kl, vl, 
                                                         pad_mask=pad_mask_l,
                                                         context_mask=context_mask)
        
        output = alpha * output_context + (1 - alpha) * output_lexicon
        
        
        #output: (batch*nh) * length_q * d_k
        #attention: (batch*nh) * length_q * length_k
        
        output = output.view(-1, self.num_head, length_q, self.d_k) #batch * nh * l_q * d_k
        output = output.permute(0, 2, 1, 3).contiguous().view(-1, length_q, self.num_head * self.d_k) #batch * l_q * (nh * d_k)
        
        output = self.norm(self.dropout(self.wo(output)) + X) #batch * l_q * d_x
        
        attention_context = attention_context.view(-1, self.num_head, length_q, length_k) #batch * nh * l_q * l_k
        attention_lexicon = attention_lexicon.view(-1, self.num_head, length_q, length_kl) #batch * nh * l_q * l_k
        
        return output, attention_context, attention_lexicon
    
    
    
class PointwiseFF(nn.Module):
    def __init__(self, d_x, d_ff, dropout=.0):
        super(PointwiseFF, self).__init__()
        self.w1 = nn.Conv1d(d_x, d_ff, 1)
        self.w2 = nn.Conv1d(d_ff, d_x, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_x)
        
        
    def forward(self, X):
        output = self.w2(F.relu(self.w1(X.transpose(1, 2)))) #dim of x: (batch, seq_length, d_x)
        output = self.dropout(output.transpose(1, 2))
        output = self.norm(output + X) #batch * seq_length * d_x
        return output
        
    
