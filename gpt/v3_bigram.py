import torch
import torch.nn as nn
from torch.nn import functional as F 

# hyperparameters
batch_size = 64 #independent seqs will process in llel
block_size = 256  #max context len for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

#corpus file input 
with open('gpt\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #set of chars that are in corpus of text
vocab_size = len(chars) #distinct chars in chars

#encoder and decoder
#creating mappings chars->int and int->chars
stoi = {ch:i for i,ch in enumerate(chars)} #string to integer
itos = {i:ch for i,ch in enumerate(chars)} #integer to string
encoder = lambda s : [stoi[c] for c in s] #encodes string chars to a list of integers
decoder = lambda l : ''.join([itos[i] for i in l]) #decodes list of integer to string

# train and test splits 
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]

#data loading - generating batches splits to train 
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)- block_size, (batch_size,))
    #generate all batches randomly the x&y with certain offset that is ix, def goes through all but in shuffle mode
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """" one head of self-attentnion"""
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd, head_size, bias=False)
        self.query=nn.Linear(n_embd, head_size, bias=False)
        self.value=nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #can see only past, wont let it see future and cheat
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        #input size (batch, time-step, channels)
        #output size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x) 
        q = self.key(x)
        #compute the attention scores (affinity)
        wei = q@k.transpose(-2,-1) *k.shape[-1]**-0.5 #(B,T,T)*(B,T,hs) -> (B,T,hs)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) #(B,T,hs)
        out = wei @ v #(B,T,T)*(B,T,hs) -> (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    """" multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        """ a linear projection layer used at the end of a multi-head attention block 
         in a Transformer model. It combines the outputs of all attention heads and 
         projects them back to the original embedding dimension of the model."""
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self   .proj(out))
        return out

class FeedForward(nn.Module):
    """" a simple linear layer followed by a non-linearty"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.Relu(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
        def forward(self, x):
            return self.net(x)

class Block(nn.Module):
    """" transformer block : communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_embd) #self attention
        self.ffwd = FeedForward(n_embd) # feed forward network
        self.ln1 = nn.LayerNorm(n_embd) # normalization layer1
        self.ln2 = nn.LayerNorm(n_embd) #nrmlztn lyr 2

    def forward(self, x):
        #residual sums shortcut aggregate layers
        x = x+self.sa(self.ln1(x)) #normalized resnet layer
        x = x+self.ffwd(self.ln2(x)) #normalized resnet layer
        return x
#-----------------------bigram model----------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb 
        logits = self.lm_head(tok_emb) #(B,T,vocab_size)

        if targets is None:
            loss = None 
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T, C) #batches, time, channels
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context 
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #get the predictions
            logits = logits[:, -1, :] #focus only the previous step
            probs = F.softmax(logits, dim=-1) #softmax probabilities (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) #sample from the distribution (B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

#pytorch adamW optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    #eval the loss on train & val sets
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},  val loss {losses['val']:.4f}")
    #sample a batch of data
    xb,yb = get_batch('train')
    #evaluate the loss
    logits, loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0]. tolist())) #decode the output of size max_new_tokens