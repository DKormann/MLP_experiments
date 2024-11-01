#%%
import matplotlib.pyplot as plt
import torch

#%%


def interpol(rand, x):
  step = x.int()
  frac = (x - step).reshape(-1, *((1,) * (len(rand.shape) -1)))
  return rand[step] * (1-frac) + rand[step + 1] * frac

def highnoise(f, dim):
  rand = torch.randn((f + 2,)*len(dim))
  
  for i, d in enumerate(dim):
    x = torch.linspace(0, f, d)

    rand = interpol(rand.transpose(i, 0), x).transpose(i,0)
  return rand

def highfrac(dim, F=[2, 4, 10, 20]): return sum([highnoise(f, dim) / f for f in F])
# %%

if __name__ == '__main__':
  sp = (10,20,30,4)
  assert highfrac(sp).shape == sp
  sp = (50,50)
  plt.imshow(highfrac(sp))
