import torch
from torch.nn.utils import prune

conv = torch.nn.Conv2d(3, 7, 1)
with torch.no_grad():
    conv.bias.copy_(torch.randn_like(conv.bias))
mask_bias = torch.tensor([1,1]+ [0]*5)
mask_weight = torch.zeros_like(conv.weight)
mask_weight[...] = mask_bias.reshape(-1,1,1,1)

m = prune.custom_from_mask(conv, name="weight", mask=mask_weight)
# m = prune.custom_from_mask(conv, name="bias", mask=mask_bias)

print(m(torch.rand(1,3,1,1)))

exit()