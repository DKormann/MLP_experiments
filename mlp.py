
import torch
from torch import nn

class Linear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(out_features, in_features)/in_features**.5 + torch.eye(out_features, in_features))
    self.bias = nn.Parameter(torch.zeros(out_features))
  def forward(self, x): return x @ self.weight.t() + self.bias

class MLP(nn.Module):
  def __init__(self, layers):
    super().__init__()
    self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(layers[:-2], layers[1:-1]))
    self.act = nn.ReLU()
    self.out = nn.Linear(layers[-2], layers[-1])
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    for layer in self.layers: x = self.dropout(self.act(layer(x)))
    return self.out(x)




def train(x, y, model, mask=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-2)
  y = y.to(device)
  x = x.to(device)

  def step():
    opt.zero_grad()
    p = model.train()(x)
    loss = ((p - y)**2)

    if mask is not None: loss = loss[mask]

    loss = loss.mean()
    loss.backward()
    opt.step()
    return loss.item()

  print(step())
  for _ in range(2000): print(f'{step()=}', end='\r')
  print()

  p = model.eval()(x)
  return p.detach().cpu()