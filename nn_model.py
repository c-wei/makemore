import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
#region ######################## BIGRAM IMPLEMENTATION ##############################
# N = torch.zeros((27,27), dtype=torch.int32)

# # 2d array to map bigrams
# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1

# P = (N+1).float()
# P /= P.sum(1, keepdim=True)
# g = torch.Generator().manual_seed(2147483647)

# out = []
# ix = 0
# while True:
#     p = P[ix]
#     ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#     out.append(itos[ix])
#     if ix == 0:
#         break

# log_likelihood = 0.0
# n = 0
# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
#         # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

# nll = -log_likelihood  # good loss function

#endregion #############################################################################

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)  # rand initialize 27 neuron weights, each neuron gets 27 inputs

(W**2).mean()

# fwd pass
xenc = F.one_hot(xs, num_classes=27).float()  #input to network w/one-hot encoding
logits = xenc @ W # predict log-counts
# softmax:
counts = logits.exp() # counts, equiv to N
probs = counts / counts.sum(1, keepdims=True) # probs for next char
loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

# backward pass
W.grad = None
loss.backward()

W.data += -0.1 * W.grad


# sample from NN model
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix==0:
        break
    print(''.join(out))


