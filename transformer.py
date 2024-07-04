import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GravitationLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.G = nn.Parameter(torch.randn(n_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        masses = torch.norm(K, dim=-1, keepdim=True)
        
        Q_expanded = Q.unsqueeze(-2)
        K_expanded = K.unsqueeze(-3)
        distances = torch.norm(Q_expanded - K_expanded, dim=-1)
        
        distances = distances + 1e-8 # basically to avoid divs by zero
        forces = self.G * masses.unsqueeze(-2) * masses.unsqueeze(-3) / (distances ** 2)
        
        forces = forces / forces.sum(dim=-1, keepdim=True)
        
        output = torch.matmul(forces, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)

class GravitationTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.gravitation = GravitationLayer(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gravitation_out = self.gravitation(x)
        x = self.norm1(x + self.dropout(gravitation_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class GravitationTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([GravitationTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.fc_out(x)


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


d_model = 64
n_heads = 4
n_layers = 2
d_ff = 128
max_seq_len = 50

sentence = "Gravity is replacing self-attention in this architecture"

tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens([sentence]), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]

tensor_input = torch.tensor([text_pipeline(sentence)]).to(torch.long)

vocab_size = len(vocab)
model = GravitationTransformer(d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len)

with torch.no_grad():
    output = model(tensor_input)

predicted_tokens = output.argmax(dim=-1)
predicted_words = [vocab.get_itos()[i] for i in predicted_tokens[0]]

print("Input sentence:", sentence)
print("Tokenized input:", tensor_input)
print("Model output shape:", output.shape)
print("Predicted tokens:", predicted_tokens)
print("Predicted words:", predicted_words)

first_layer = model.layers[0].gravitation
Q = first_layer.W_q(model.embedding(tensor_input))
K = first_layer.W_k(model.embedding(tensor_input))
masses = torch.norm(K, dim=-1, keepdim=True)
distances = torch.norm(Q.unsqueeze(-2) - K.unsqueeze(-3), dim=-1)
forces = first_layer.G[0] * masses.unsqueeze(-2) * masses.unsqueeze(-3) / (distances ** 2 + 1e-8)
normalized_forces = forces / forces.sum(dim=-1, keepdim=True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(normalized_forces[0, 0].detach().numpy(), annot=True, fmt='.2f', cmap='viridis')
plt.title('Gravity-like weights for the first head in the first layer')
plt.xlabel('Key positions')
plt.ylabel('Query positions')
plt.show()

print("Learned gravitational constants:")
print(first_layer.G.detach().numpy())