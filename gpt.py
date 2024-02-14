import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
embed_dim = 32

torch.manual_seed(1337)

# read the data
data_file = "data/tinyshakespeare.txt"
with open(data_file, "r", encoding="utf-8") as file:
    text = file.read()

# all the unique characters vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create character to index mapping
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# train and val split
data = torch.tensor(encode(text), dtype=torch.long)
tn = int(len(data) * 0.9)
train_data, val_data = data[:tn], data[tn:]

# data loader
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        total_loss = 0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total_loss += loss.item()
        out[split] = total_loss / eval_iters
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.scale_factor = head_dim ** 0.5

    def forward(self, input, attn_mask=None):
        _, T, _ = input.shape
        queries = self.query(input)
        keys = self.key(input)
        values = self.value(input)

        # calculate attention scores
        scores = (queries @ keys.transpose(-2, -1)) / self.scale_factor
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:T, :T] == 0, float("-inf"))

        # perform the weighted aggregation of the values
        probs = F.softmax(scores, dim=-1)
        out = probs @ values
        return out

class MultiHead(nn.Module):
    def __init__(self, head_num, head_dim):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_dim) for _ in range(head_num)])

    def forward(self, input, attn_mask=None):
        return torch.cat([head(input, attn_mask) for head in self.heads], dim=-1)

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(block_size, embed_dim)
        self.mask = torch.tril(torch.ones((block_size, block_size), device=device))
        self.self_attn_head = MultiHead(4, embed_dim // 4)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape

        # idx and targets are both (B,T) tensor
        token_emb = self.token_embedding(idx)  # (B,T,C)
        pos_enc = self.positional_encoding(torch.arange(T, device=device))  # (T,C)
        token_emb = token_emb + pos_enc  # (B,T,C)
        attn = self.self_attn_head(token_emb, self.mask)  # (B,T,C)
        logits = self.lm_head(attn)  # (B,T,vocab_size)

        # calculate the loss
        if targets is not None:
            _, _, C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_token):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_token):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B,C)
            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


# create the model and optimizer
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    xb, yb = get_batch("train")

    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

    # evaluate the loss on train and val data
    if iter % eval_interval == 0:
        out = estimate_loss(model)
        print(f"iter {iter:4d}, train loss: {out['train']:.4f}, val loss: {out['val']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
