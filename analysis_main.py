import argparse
from collections import defaultdict
import glob
import itertools
import pathlib
from typing import Any, List, Tuple, Union, Dict

import torch
from torchvision import transforms as T
from PIL import Image
import wandb
import pandas as pd

from ProbabilityAwarePruningStyleTransfer import ElasticStyleTransfer
import pruning_utils
import losses
import plot_utils
import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

__version__ = "0.2.add_plotly"

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
    default=[.0, .2, .4, .6, .8],
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
  base_config = dict(
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
    name=base_config["__version__"],
    config=base_config,
  )

  all_code = []
  style_tensor_list = []
  for style_path in base_config["style_paths"]:
    with torch.no_grad():
      style_tensor = EST.style_preprocess(path2tensor(style_path))
      style_tensor_list.append(style_tensor)
      all_code.append(EST.SP(style_tensor))
    
  all_code = torch.concat(all_code)

  ## Nested by 
  # layer_idx
  # content_path, mode, pruning_rate, prune_from_smaller
  # style_path
  for layer_idx in base_config["layers"]:
    current_dict = layer_config = dict(
      layer_idx = layer_idx,
    )

    df_dict = defaultdict(list)

    for content_path, mode, pruning_rate, prune_from_smaller in itertools.product(
      base_config["content_paths"],
      base_config["modes_list"],
      base_config["pruning_rate_list"],
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
      
      current_dict = log_dict = dict(
          pruning_rate=pruning_rate,
          content_path=content_path,
          mode=mode,
          from_smaller_flag=prune_from_smaller,
        )
      current_dict.update(
        layer_config
      )

      content_losses = losses.content_loss(result, content_tensor).cpu().numpy()
      style_losses = losses.style_loss(result, style_tensor).cpu().numpy()
      for ith_style, style_path, in  enumerate(
        base_config["style_paths"],
      ):
        current_dict = loss_dict = dict(
          style_path=style_path,
          content_loss=content_losses[ith_style],
          style_loss=style_losses[ith_style],
        )
        loss_dict.update(log_dict)

        if args.save_image_flag:
          images = dict(
            stylized = tensor2wandb_img(result[ith_style]),
            style = tensor2wandb_img(style_tensor_list[ith_style]),
            content = tensor2wandb_img(content_tensor[ith_style]),
          )
          current_dict.update(
            "images", images
          )
      
        for k, v in current_dict.items():
          if not isinstance(v, dict):
            df_dict[k].append(v)
        
        wandb.log(current_dict)

        pp({k:val for k, val in filter(lambda x: not isinstance(x[1], dict), current_dict.items())})
    df = pd.DataFrame(df_dict)
    wandb.log(
      {
        f"table_layer_{layer_idx}": df,
        **{
          f"plotly_{fig_notation}":fig
          for fig_notation, fig in plot_utils.get_plotly_fig(
            df,
            title_prefix=f"layer_idx_{layer_idx}",
          ).itmes()
        }
      }
    )

    # TODO : tqdm something
    # TODO : postorder something
    
  run.finish()
    
    
    


if __name__ == "__main__":
  with torch.no_grad():
    main()
