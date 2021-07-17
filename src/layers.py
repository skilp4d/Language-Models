import torch
import math
from torch import nn, optim
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttentionLayer(nn.Module):
    def __init__(self, emb_dimension, n_head, dropout):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.n_head = n_head
        self.qw = nn.Linear(self.emb_dimension, self.emb_dimension)
        self.kw = nn.Linear(self.emb_dimension, self.emb_dimension)
        self.vw = nn.Linear(self.emb_dimension, self.emb_dimension)
        self.proj = nn.Linear(self.emb_dimension, self.emb_dimension)
        self.dropout = nn.Dropout()
        
    def forward(self, x, mask=None):
        B, T, V = x.shape
        
        q = self.qw(x).view(B, T, self.n_head, V // self.n_head).transpose(1, 2) # shape will be B, b_head, T, V // n_head
        k = self.kw(x).view(B, T, self.n_head, V // self.n_head).transpose(1, 2)
        v = self.vw(x).view(B, T, self.n_head, V // self.n_head).transpose(1, 2)
        
        
        # scaled dot attention
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1))) # shape will be B, n_head, T, T
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, T, 1).unsqueeze(1)
            att.masked_fill_(mask, -1e9)
        
        att = torch.softmax(att, -1)
        out = att @ v
        att = self.dropout(att)
        
        out = out.transpose(1, 2)
        out = out.reshape(B, T, V)
        out = self.proj(out)
        out = self.dropout(out)
        
        return {
            'output': out,
            'mask': mask,
            'attention': att
        }
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_dimension=512, n_head=8, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dimension)
        self.ln2 = nn.LayerNorm(emb_dimension)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttentionLayer(emb_dimension, n_head, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dimension, 4*emb_dimension),
            nn.GELU(),
            nn.Linear(4*emb_dimension, emb_dimension),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        x = x + self.self_attention(self.ln1(x), mask)['output']
        x = x + self.feed_forward(self.ln2(x))
        return x