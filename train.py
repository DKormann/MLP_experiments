#%%
import torch
import matplotlib.pyplot as plt
from noise import highfrac
import numpy as np
import mlp

device = 'cuda' if torch.cuda.is_available() else 'mps'

#%%
xdim = 3
ydim = 4

n_sample = 1000
resol = int(round(n_sample ** (1/xdim)))

y = torch.stack([highfrac((resol,) * xdim) for _ in range(ydim)]).permute(*range(1, xdim+1), 0)
x = torch.tensor(np.mgrid.__getitem__([slice(resol) for _ in range(xdim)]))/resol
x = x.permute([*range(1, x.dim()), 0])

half = resol//2
y[half:] = y[:-half].flip(0)

#%%

x.shape, y.shape
t = torch.linspace(0, 1, resol)

a,b = 0.65, 0.8
halfmask = (t < a).logical_or(t > b)
fullmask = (t < (1-b)).logical_or(t > (1-a)).logical_and(halfmask)


#%%

for case in range(2):

  torch.manual_seed(77)
  model = mlp.MLP([xdim, 100, 100, ydim])
  mask = halfmask if case else fullmask
  print('# halfmask' if case else '# fullmask')

  p = mlp.train(x, y, model, mask)

  def eval():
    ev = model.eval()(x)-y
    ev = ev[~halfmask]
    ev = ev.square().mean().item()
    return ev

  print(f'{eval()=}')

  if xdim == 1:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, d in zip([0,1], [y, p.detach()]):
      axs[i].plot(d.cpu())
      axs[i].axvspan(a*resol, b*resol, alpha=0.2, color='red')
      if not case: axs[i].axvspan(resol-a*resol, resol-b*resol, alpha=0.2, color='red')
  else:
    K = min(2, ydim)
    fig, axs = plt.subplots(K, 2, figsize=(6, 3*K))
    for k in range(K):
      for i, d in zip([0,1], [y.flatten(2,-2), p.detach().flatten(2,-2)]):
        axs[k,i].imshow(d[:, :,0, k].cpu())
        axs[k,i].axhline(a*resol, color='red')
        axs[k,i].axhline(b*resol, color='red')
        if not case:
          axs[k,i].axhline(resol-a*resol, color='red')
          axs[k,i].axhline(resol-b*resol, color='red')
    pass
  plt.show()

  #%%