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

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensor
        logits = self.token_embedding_table(idx)  # (B,T,C)
        logits = logits.view(-1, self.token_embedding_table.embedding_dim)
        targets = targets.view(-1)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_token):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_token):
            # get the predictions
            logits = self.token_embedding_table(idx)
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
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # evaluate the loss on train and val data
    if iter % eval_interval == 0:
        out = estimate_loss(model)
        print(f"iter {iter:4d}, train loss: {out['train']:.2f}, val loss: {out['val']:.2f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
