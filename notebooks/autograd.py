# %%
import torch
x = torch.tensor([3, 10], requires_grad=True, dtype=float)
y = 0.5 * x[0] + 1.3 * x[1]
grad = torch.autograd.grad(y, x)
grad
# %%
