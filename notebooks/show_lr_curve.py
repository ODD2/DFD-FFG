# %%
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR
# %%
param = torch.nn.Parameter(torch.randn(10))
opt = torch.optim.AdamW([param], lr=1e-3)
# lr = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim,
#     2,
#     1e-4
# )

# lr = torch.optim.lr_scheduler.ConstantLR(optim, factor=0.5, total_iters=4)

lr = LinearLR(opt, start_factor=2, total_iters=4)

lrs = []
x = [i for i in range(10)]
for _ in x:
    lrs.append(lr.get_last_lr())
    lr.step()
plt.plot(x, lrs)
plt.show()

# %%
