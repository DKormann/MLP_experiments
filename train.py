
#%%
import torch
from torch import nn
import matplotlib.pyplot as plt
from noise import linnoise

device = 'cuda' if torch.cuda.is_available() else 'mps'

n = 1000

F = [0.5, 0.2, 0.1]
x = torch.linspace(0,1,1000).unsqueeze(1).to(device)
y = sum([linnoise(n//2, int(f*100)) * f for f in F]).unsqueeze(1).to(device) * 0.1
y += x[:n//2]
y = torch.concat([y, y[torch.arange((n//2)-1, -1, -1)]], dim=0)

plt.plot(x.cpu(), y.cpu())

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
    self.inp = nn.Linear(1, layers[0])
    self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(layers[:-1], layers[1:]))
    self.act = nn.ReLU()
    self.out = nn.Linear(layers[-1], 1)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = self.inp(x)
    for layer in self.layers: x = self.dropout(self.act(layer(x)))
    return self.out(x)

#%%
model = MLP([200, 100, 40])
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

#%%

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
plt.plot(x.cpu(), y.cpu())
plt.plot(x.cpu(), p.detach().cpu())


#%%

for p in model.parameters():
  print(p.requires_grad, p.device)