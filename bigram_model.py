# take one text file as input and generate more things like it
import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27,27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# 2d array to map bigrams
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# visualization of 2d array
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i,j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()

# probability of drawing the index number based on frequency of appearance from the 2d array
# g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(itos[ix])

P = (N+1).float()
P /= P.sum(1, keepdim=True)
g = torch.Generator().manual_seed(2147483647)

out = []
ix = 0
while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        break
print(''.join(out))

log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

nll = -log_likelihood  # good loss function