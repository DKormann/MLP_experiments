#%%
import matplotlib.pyplot as plt
import torch



def linnoise(n, f):

  rand = torch.randn(n//f + 1)
  x = torch.arange(0, n/f, 1/f)
  step = x.int()
  frac = x - step

  res = rand[step]
  next = rand[step + 1]
  interp = res * (1-frac) + next * frac
  return interp




plt.scatter(torch.arange(20), linnoise(20,2))
