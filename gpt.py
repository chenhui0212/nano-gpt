import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
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

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor
        token_emb = self.token_embedding(idx)  # (B,T,C)
        logits = self.lm_head(token_emb)  # (B,T,vocab_size)

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
            # get the predictions
            logits, _ = self(idx)
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
        print(f"iter {iter:4d}, train loss: {out['train']:.2f}, val loss: {out['val']:.2f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
