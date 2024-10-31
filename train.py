
#%%
import torch
from torch import nn
import matplotlib.pyplot as plt
from noise import highfrac, highnoise

device = 'cuda' if torch.cuda.is_available() else 'mps'
import numpy as np

#%%
xdim = 2
ydim = 4

n_sample = 1000
resol = int(round(n_sample ** (1/xdim)))

y = torch.stack([highfrac((resol,) * xdim) for _ in range(ydim)]).permute(*range(1, xdim+1), 0)
x = torch.tensor(np.mgrid.__getitem__([slice(resol) for _ in range(xdim)]))/resol
x = x.permute([*range(1, x.dim()), 0])

y = y.flatten(0, -3).to(device)
x = x.flatten(0, -3).to(device)

x.shape, y.shape

#%%
class Linear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(
      torch.randn(out_features, in_features)/in_features**.5 + torch.eye(out_features, in_features))
    self.bias = nn.Parameter(torch.zeros(out_features))

  def forward(self, x): return x @ self.weight.t()# + self.bias

class MLP(nn.Module):
  def __init__(self, layers):
    super().__init__()
    # self.inp = nn.Linear(layers[0], layers[1])
    self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(layers[:-2], layers[1:-1]))
    self.act = nn.ReLU()
    self.out = nn.Linear(layers[-2], layers[-1])
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    # x = self.inp(x)
    for layer in self.layers: x = self.dropout(self.act(layer(x)))
    return self.out(x)

#%%
model = MLP([xdim, 200, 100, 40, ydim])
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

assert model(x).shape == y.shape

#%%

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
def step():
  model.train()
  opt.zero_grad()
  p = model(x)
  loss = ((p - y)**2).mean()
  loss.backward()
  opt.step()
  return loss.item()

print(step())
for i in range(2000): print(step(), end='\r')

# %%
p = model.eval()(x)
print(p.shape)

#%%

k = 0
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(y[:, :, k].cpu())
axs[1].imshow(p[:, :, k].detach().cpu())

plt.show()