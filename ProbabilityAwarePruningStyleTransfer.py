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
    
  def forward(self, content, style=None, code=None):
    assert (style is not None) or (code is not None)
    assert not ((style is not None) and (code is not None))
    
    assert content.shape[0] == 1, "only support 1 content image now"

    G, SP = self.G, self.SP
    if code is None:
      code = SP(
        self.style_preprocess(style)
      )
    content = content.repeat(code.shape[0], *[1]*(content.dim()-1))

    output = G(content, code)

    return output

  @property
  def pruneable_modules(self):
    ret = OrderedDict([
      *filter(
        lambda x:hasattr(x[1],"instancenorm"),
        [(i, layer) for i, layer in enumerate(self.G.layers[:-1])]
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

  def prune_ith_layer(
    self, ith, remain_channels=.5,
    mode="bias/abs(scale)",
    importance_scores=None,
    **kwargs,
  ):
    module = self.pruneable_modules[ith]
    sub_module_name = {
      ResidualBlock : "relu", # conv2 is not considered this time
      UpsampleConvInRelu : "activation",
    }[type(module)]

    sub_module = module.get_submodule(sub_module_name)

    if len(sub_module._forward_hooks) > 0:
      raise ValueError("hook exist")

    if mode == "L1":
      parameter_name = {
        ResidualBlock : "conv1.weight",
        UpsampleConvInRelu : "conv.weight"
      }[type(module)]
      para = module.get_parameter(parameter_name)
      dims = list(range(para.dim()))
      dims.remove(0)
      importance_scores = para.abs().mean(dims)
      
    
    hook = ProbablityAwarePruningHook(
      remain_channels=remain_channels,
      mode=mode,
      importance_scores=importance_scores,
      **kwargs,
    )
    handle = sub_module.register_forward_hook(hook)
    self.pruning_handles.append(handle)

    return handle


if __name__ == "__main__":
  with torch.no_grad():
    T = ElasticStyleTransfer()
    
    T.prune_ith_layer(3,remain_channels=120)
    T.G.layers[3].relu.register_forward_hook(lambda cls, input, output: print(output.mean(dim=(2,3))))
    content = torch.rand(1, 3, 128, 128)
    style = torch.rand(3, 3, 128, 128)
    code = T.SP(T.style_preprocess(style))
    inner_scale = T.G.layers[3].fc_gamma1(code)
    inner_bias = T.G.layers[3].fc_beta1(code)
    score = inner_bias/(inner_scale.abs()+1e-5)*-1

    print("score")
    print(score)
    print("inner_scale")
    print(inner_scale)
    print("inner_bias")
    print(inner_bias)

    print(torch.topk(score, dim=1, k=8))
    T(content, style)
    T.unhook_all()

  print("done")
