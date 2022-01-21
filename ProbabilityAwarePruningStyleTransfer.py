from typing import OrderedDict

import torch
import torchvision.transforms as T
from torch import nn

from pruning_utils import ProbablityAwarePruningHook, available_modes
from utils.stylePredictor26 import pretrainedStylePredictor
from utils.transform26 import (ResidualBlock, UpsampleConvInRelu,
                               pretrainedGhiasi)


class ElasticStyleTransfer(nn.Module):
  def __init__(self):
    super(ElasticStyleTransfer, self).__init__()
    self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
    self.SP = pretrainedStylePredictor().to(device).eval()
    self.G = pretrainedGhiasi().to(device).eval()

    self.style_preprocess = torch.nn.Sequential(
      T.Resize(256,),
      T.CenterCrop(256),
    )
    
    self._pruning_handles = []
    
  def forward(self, content, style, code=None):
    G, SP = self.G, self.SP
    if code is None:
      code = SP(
        self.style_preprocess(style)
      )

    output = G(content, code)

    return output

  @property
  def pruneable_modules(self):
    ret = OrderedDict([
      *filter(
        lambda x:hasattr(x[1],"instancenorm"),
        [(i, layer) for i, layer in enumerate(self.G.layers)]
      )
    ])
    return ret

  @property
  def pruning_handles(self):
    return self._pruning_handles
  
  def unhook_all(self):
    for handle in self._pruning_handles:
      handle.remove()
    self._pruning_handles = []

  def prune_ith_layer(self, ith, channels=.5, importance_scores=None):
    module = self.pruneable_modules[ith]
    hook = ProbablityAwarePruningHook(
      remain_channels=channels,
      importance_scores=importance_scores,
    )
    if len(module._forward_hooks) > 0:
      raise ValueError("hook exist")

    sub_module_name = {
      ResidualBlock : "conv1", # conv2 is not considered this time
      UpsampleConvInRelu : "conv",
    }[type(module)]
    handle = module.get_submodule(sub_module_name).register_forward_hook(hook)
    self.pruning_handles.append(handle)

    return handle


if __name__ == "__main__":
  T = ElasticStyleTransfer()
  
  T.prune_ith_layer(3)
  T.unhook_all()

  print("done")
