from turtle import forward
import onnx
import onnxruntime

import torch
from ProbabilityAwarePruningStyleTransfer import ElasticStyleTransfer
from utils.transform26 import (ResidualBlock, UpsampleConvInRelu,
                               pretrainedGhiasi)

from collections import OrderedDict
import tempfile

mode_list = [
  "L1", "bias/abs(scale)", "abs(scale)", "bias", "from_first_layers",
]


class new_SP:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.G = pretrainedGhiasi()
    self.feed_examples = OrderedDict()
    self.feed_examples["mode_idx"] = torch.tensor(1,dtype=torch.int, device=self.device)
    self.feed_examples["prune_from_smaller"] = torch.tensor(1.0, device=self.device)
    self.feed_examples["input_image"] = torch.rand((1,3,384,384), device=self.device)
    # register
    # for layer_idx in ... :self.feed_examples[f"{prefix}_keep_rate"] = torch.tensor(.6, dtype=torch.float32)
    self.rate2idx = lambda tensor:(tensor/.199-1).long()

    self.hook_handles = []

  def hook_for_module(
    self,
    module:torch.nn.Module,
    prefix:str,
  ):
    submodule_names_dict = {
        ResidualBlock : {
            "conv":"conv1",
            "scale" : "fc_gamma1",
            "bias" : "fc_beta1",
        },
        UpsampleConvInRelu:{
            "conv":"conv",
            "scale" : "fc_gamma",
            "bias" : "fc_beta",
        }
    }.get(type(module), None)
    
    assert submodule_names_dict is not None
    submodules_dict = {
        key : module.get_submodule(submodule_name)
        for key, submodule_name in submodule_names_dict.items()
    }
    
    self.feed_examples[f"{prefix}_keep_rate"] = (
      torch.tensor(.6, dtype=torch.float32)
    )

    channels = submodules_dict["conv"].out_channels
    rate_to_remain_channels_number = {
      rate : int(rate*channels)
      for rate in [.2, .4, .6, .8, 1.]
    }

    # "L1", "bias/abs(scale)", "abs(scale)", "bias", "from_first_layers",
    _score_dict = {
      "L1" : (
        submodules_dict["conv"]
        .weight.abs().mean(axis=(1,2,3))
        .view(1,-1,1,1)
      ),
      "bias/abs(scale)": None,
      "abs(scale)":None,
      "bias":None,
      "from_first_layers":torch.arange(
        channels,
        device=self.device,
      ).view(1,-1,1,1),
    }
    
    
    _temp_dict = {} # store scale, bias
    def hook_scale(cls, input, output):
      _temp_dict["abs(scale)"] = output.abs()
      
    def hook_bias(cls, input, output):
      _temp_dict["bias"] = output
      
    def hook_conv(cls, input, output):
      eps = 1e-5
      _score_dict.update(_temp_dict)
      _score_dict["bias/abs(scale)"] = _score_dict["bias"]/(_score_dict["abs(scale)"]+eps)

      _score_tensor = (
        torch.stack([_score_dict[key] for key in mode_list])[self.feed_examples["mode_idx"]]
      * self.feed_examples["prune_from_smaller"]-0.5
      )

      rate_to_masks = [
        (
          torch.zeros(channels)
          .scatter_(
            0,
            torch.topk(_score_tensor.view(-1),remain_channels_number)[1],
            1.
          )
        ).view(1,-1,1,1)
        for rate, remain_channels_number in rate_to_remain_channels_number.items()
      ]
      mask = torch.stack(rate_to_masks)[
        self.rate2idx(self.feed_examples["rate"])
      ]

      return mask*output
    
    return {
      "hook_scale" : submodule_names_dict["scale"].register_forward_hook(hook_scale),
      "hook_bias" : submodule_names_dict["bias"].register_forward_hook(hook_bias),
      "hook_conv" : submodule_names_dict["conv"].register_forward_hook(hook_conv),
    }


  def hook_all(self): 
    G = self.G

    for ith_layer, module in enumerate(G.layers[:-1]):
      if hasattr(module, "instancenorm"):
        print(ith_layer)
      
        
        
        
  def forward(self, x):
    for layer in self.G.layers:
      x = layer(x)
    return torch.sigmoid(x)



def main():
  # just support batch_size : 1
  G = pretrainedGhiasi()

  feed_examples = dict(
    mode_idx = torch.tensor(1,dtype=torch.int),
  )

  # G.layers[:-1] : not include output layer
  for ith_layer, module in enumerate(G.layers[:-1]):
    if hasattr(module, "instancenorm"):
      print(ith_layer)

      prefix = {
        ResidualBlock:"Res",
        UpsampleConvInRelu:"UpConv",
      }.get(type(module))
      
      name_for_onnx = f"{prefix}_{ith_layer}_prune_mask"
      



      

if __name__=="__main__":
  main()