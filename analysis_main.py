import argparse
from collections import defaultdict
import glob
import itertools
import pathlib
from typing import Any, List, Tuple, Union

import torch
from torchvision import transforms as T
from PIL import Image
import wandb
import pandas as pd

from ProbabilityAwarePruningStyleTransfer import ElasticStyleTransfer
import pruning_utils
import losses
import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

__version__ = "0.1"

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, dictstyle_df_dict: Any) -> None:
        self[name] = dictstyle_df_dict

    def __delattr__(self, name: str) -> None:
        del self[name]

def parser_fun():
  parent_folder = pathlib.Path(__file__).parent

  parser = argparse.ArgumentParser(
    description="The main script to analysis",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-p", "--pruning_rate_list",
    help="grid of pruning rate list",
    nargs="+",
    default=[0, .2, .4, .6, .8],
    type=float,
  )

  parser.add_argument(
    "-s", "--style_template",
    help="style images template",
    default=f"{parent_folder}/style_images/*.jpg",
  )
  
  parser.add_argument(
    "-c", "--content_template",
    help="content images template",
    default=f"{parent_folder}/other_content/*.jpg",
  )

  parser.add_argument(
    "-m", "--modes_list",
    help="pruning modes list",
    nargs="+",
    default=["L1"]+pruning_utils.available_modes,
  )

  parser.add_argument('--save-image', dest='save_image_flag', action='store_true')
  parser.add_argument('--no-save-image', dest='save_image_flag', action='store_false')
  parser.set_defaults(save_image_flag=False)

  args = parser.parse_args()
  pp("args : ")
  pp(args)
  return args


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  EST = elastic_model = ElasticStyleTransfer().to(device)

  args = parser_fun()
  base_config = EasyDict(
    layers = list(EST.pruneable_modules.keys()),
    style_paths = glob.glob(args.style_template),
    pruning_rate_list = args.pruning_rate_list,
    content_paths = glob.glob(args.content_template),
    modes_list = args.modes_list,
    __version__ = __version__,
  )

  pp(base_config)
  
  path2tensor = lambda path : T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
  tensor2wandb_img = lambda tensor : wandb.Image(T.ToPILImage()(tensor[0].cpu()))

  run = wandb.init(
    project="ElascitPruningStyleTransfer",
    entity="mistake0316",
    name=base_config.__version__,
    config=base_config,
  )

  all_code = []
  for style_path in base_config.style_paths:
    with torch.no_grad():
      style_tensor = EST.style_preprocess(path2tensor(style_path))
      all_code.append(EST.SP(style_tensor))
    
  all_code = torch.concat(all_code)

  for layer_idx in base_config.layers:
    layer_config = EasyDict(
      layer_idx = layer_idx,
    )

    df_dict = defaultdict(list)

    for content_path, mode, pruning_rate, prune_from_smaller in itertools.product(
      base_config.content_paths,
      base_config.modes_list,
      base_config.pruning_rate_list,
      [True, False],
    ):
      EST.unhook_all()
      EST.prune_ith_layer(
        ith=layer_idx,
        remain_channels=1-pruning_rate,
        mode=mode,
        prune_from_smaller=prune_from_smaller,
      )
      with torch.no_grad():
        content_tensor = path2tensor(content_path)
        H, W = content_tensor.shape[2:]
        content_tensor = content_tensor[:, :, :H-H%32, :W-W%32]
        
        result = EST.forward(
          content_tensor,
          code=all_code,
        )
      
      log_dict = dict()
      log_dict.update(
        layer_config
      )
      log_dict.update(
        dict(
          pruning_rate=pruning_rate,
          content_loss=losses.content_loss(result, content_tensor).item(),
          style_loss=losses.style_loss(result, style_tensor).item(),
          style_path=style_path,
          content_path=content_path,
          mode=mode,
          from_smaller_flag=prune_from_smaller,
        )
      )
      
      if args.save_image_flag:
        images = dict(
          stylized = tensor2wandb_img(result),
          style = tensor2wandb_img(style_tensor),
          content = tensor2wandb_img(content_tensor),
        )
        log_dict.update(
          "images", images
        )
      
      for k, v in log_dict.items():
        if not isinstance(v, dict):
          df_dict[k].append(v)
      
      
      wandb.log(log_dict)
      pp({k:array[-1] for k, array in df_dict.items()})
    style_df = pd.DataFrame(df_dict)
    wandb.log({f"table_{layer_idx}": df_dict})

    # TODO : add plotly for df
    # TODO : tqdm something
    # TODO : postorder something
    # TODO : batchfiy style code
    
  run.finish()
    
    
    


if __name__ == "__main__":
  with torch.no_grad():
    main()
