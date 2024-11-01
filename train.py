#%%
import torch
import matplotlib.pyplot as plt
from noise import highfrac
import numpy as np
import mlp

device = 'cuda' if torch.cuda.is_available() else 'mps'

#%%
xdim = 1
ydim = 4

n_sample = 1000
resol = int(round(n_sample ** (1/xdim)))

y = torch.stack([highfrac((resol,) * xdim) for _ in range(ydim)]).permute(*range(1, xdim+1), 0)
x = torch.tensor(np.mgrid.__getitem__([slice(resol) for _ in range(xdim)]))/resol
x = x.permute([*range(1, x.dim()), 0])

half = resol//2
y[half:] = y[:-half].flip(0)

x.shape, y.shape
t = torch.linspace(0, 1, resol)

side_mask = 0.7
center_mask = 0.5

#%%
for case in range(2):
  torch.manual_seed(77)
  model = mlp.MLP([xdim, 100, 100, 100, ydim])
  mask_center = side_mask if case else center_mask

  print('# centermask' if case else '# sidemask')

  r = 0.1
  mask = (t < mask_center - r).logical_or(t > mask_center + r)

  p = mlp.train(x, y, model, mask, epochs=5000)

  def eval():
    ev = model.eval()(x)-y
    return ev[~mask].square().mean().item()

  print(f'{eval()=}')

  if xdim == 1:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, d in zip([0,1], [y, p.detach()]):
      axs[i].plot(d.cpu())
      axs[i].axvspan((mask_center-r)*resol, (mask_center+r)*resol, alpha=0.2, color='red')
  else:
    K = min(2, ydim)
    fig, axs = plt.subplots(K, 2, figsize=(6, 3*K))
    for k in range(K):
      for i, d in zip([0,1], [y.flatten(2,-2), p.detach().flatten(2,-2)]):
        axs[k,i].imshow(d[:, :,0, k].cpu())
        axs[k,i].axvspan((mask_center-r)*resol, (mask_center+r)*resol, alpha=0.2, color='red')

  plt.show()

#%%

plt.plot(p.detach().cpu())
plt.plot(y.cpu())
plt.axvspan((mask_center-r)*resol, (mask_center+r)*resol, alpha=0.2, color='red')
