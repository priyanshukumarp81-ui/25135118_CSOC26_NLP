# %%
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# %%
class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,input):
        return self.embedding(input)*math.sqrt(self.d_model)  

# %%
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (-math.log(10000.0) / d_model))
       
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)    
        

# %%
class CausalMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads,dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = attention @ V
        output = output.transpose(1, 2)
        output = output.contiguous().view(B, T, self.d_model)
        output = self.out(output)

        return output    

# %%
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        d_ff = 4*d_model
        self.linear1 = nn.Linear(d_model,d_ff)
        
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,X):
        X = self.linear1(X)
        X = F.gelu(X) 
        X = self.dropout(X)
        X = self.linear2(X)
        
        return X 

# %%
class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        
        self.ln1=nn.LayerNorm(d_model)
        self.attention = CausalMultiHeadAttention(d_model=d_model,num_heads=num_heads,dropout=0.1)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,X):
        attention_out = self.attention(self.ln1(X))
        X = X + self.dropout(attention_out)
        
        ffn_out = self.ffn(self.ln2(X))
        X = X + self.dropout(ffn_out)
        
        return X    
        

# %%
class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,seq_len,num_heads,num_layer,dropout):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size=vocab_size,d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model,seq_len,dropout)
        self.decoder_block = nn.ModuleList([DecoderBlock(d_model,num_heads,dropout) for _ in range(num_layer)])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)
        self.lm_head.weight = self.input_embedding.embedding.weight
        
    def forward(self,X):
        X = self.input_embedding(X)
        X = self.position_embedding(X)
        for dec_block in self.decoder_block:
            X = dec_block(X)
        X = self.ln_final(X)
        logits = self.lm_head(X)
        
        return logits        


