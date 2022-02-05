from collections import defaultdict
import itertools
from operator import itemgetter
from typing import Any, List, Tuple, Union, Dict

from torchvision import transforms as T
from PIL import Image
import plotly.graph_objs as go
import pandas as pd
from wandb import Plotly

from ProbabilityAwarePruningStyleTransfer import ElasticStyleTransfer
import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

import plotly.express as px
import os
from typing import Any, List, Tuple, Union, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots

__all__ = [
  "get_plotly_fig", 
  "default_color_seed",
]

default_color_seed = 282886



def get_plotly_fig(
  df,
  title_prefix=None,
  )->Dict:
  df = df.copy()
  df["content_name"]=df["content_path"].apply(os.path.basename)
  df["style_name"]=df["style_path"].apply(os.path.basename)
  
  df.update(
    df.groupby(
      by=[
        "mode",
        "from_smaller_flag",
        "content_name",
        "style_name"
      ]).transform(
      # minus unprune content/style loss
      lambda series:series-series.values[0]
    )
  )

  df["plotly_group"] = pd.DataFrame([
    df["mode"].apply(str),
    df["from_smaller_flag"].apply(
      {
        True:"prune_from_smaller",
        False:"prune_from_bigger",
      }.get
    )
  ]).agg("-".join)

  # https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function/20233047
  def quantile(q):
    def inner(series):
      return series.quantile(q)
    inner.__name__ = f"quantile_{q:02f}"
    return inner

  # https://stackoverflow.com/a/43897124
  quantile_levels = [.25, .5, .75] # __len__ must be odd
  df_for_plotly = df.groupby(["pruning_rate","plotly_group"]).agg(
    **{
      f"{loss_name}$quantile_{q:.02f}":
      (loss_name, quantile(q=q))
      for loss_name in ["content_loss", "style_loss"]
      for q_idx, q in enumerate(quantile_levels)
    }
  ).reset_index()
  figs = dict()

  figs = {
    y:
    get_quantile_bands(
      df_for_plotly,
      y=y,
      group="plotly_group",
    )
    for y in ["content_loss", "style_loss"]
  }

  for fig_name, fig in figs.items():
    text = fig_name
    if title_prefix is not None : text = "$".join([title_prefix, text])
    fig.update_layout(
      title=text,
    )
  
  return figs

def get_quantile_bands(df, y="content_loss", x="pruning_rate", group="plotly_group", seed=None):
  np.random.seed(default_color_seed if seed is None else seed)
  
  df = df.copy()
  df_grouped = df.groupby(group)
  
  objs = []
  for group_name, local_df in df_grouped:
    series_x = local_df[x]
    _local_df = local_df.filter(regex=f"^{y}")
    assert _local_df.values.shape[1]%2
    
    _mid_idx = _local_df.values.shape[1]//2

    _color_unasign_alpha = "rgba({},{},{},{{}})".format(*np.random.randint(0,256,3))
    color = _color_unasign_alpha.format(1.0)
    fill_color = _color_unasign_alpha.format(.05)
    #https://plotly.com/python/continuous-error-bars/

    for col_idx, (col_name, values)in enumerate(_local_df.iteritems()):
      __obj = go.Scatter(
        name=col_name.split("$")[-1],
        x=series_x,
        y=values,
        legendgroup=group_name,# https://plotly.com/python/legend/
        legendgrouptitle_text=group_name,
        marker=dict(color=color),
        line=dict(width=0),
        mode='lines',
        fillcolor=fill_color,
        fill='tonexty',
        showlegend=False,
      )
      if col_idx == 0:
         __obj.update(
          fill=None,
          fillcolor=None,
         )
      elif col_idx == _mid_idx:
        __obj.update(
          line=dict(width=None, color=color),
          showlegend=True,
        )

      objs.append(
        __obj
      )
  
  fig = go.Figure(
    objs,
  )
  ## it remains bug...
  ## it does not display with hivertemplate with ``legendgrouptitle_text`` has value
  ## fig.update_traces(
  ##   hovertemplate="%{y:+.3e}",
  ## )
  fig.update_layout(
    legend=dict(
      itemwidth=30,
      itemsizing="constant",
    ),
    hovermode="x unified",
    yaxis_title=y,
    xaxis_title=x,
    hoverlabel=dict(align="right"),
  )
  return fig

if __name__ == "__main__":
  from analysis_main import parser_fun
  args = parser_fun()

  # generate sample dataframe in analysis_main
  base_config = dict(
    layers = [3,5,6],
    style_paths = [f"S_{i}" for i in range(5)],
    pruning_rate_list = [0.2 * rate for rate in range(5)],
    content_paths = [f"C_{i}" for i in range(5)],
    modes_list = args.modes_list,
    __version__ = "__plot__",
  )

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
      current_dict = log_dict = dict(
          pruning_rate=pruning_rate,
          content_path=content_path,
          mode=mode,
          from_smaller_flag=prune_from_smaller,
      )
      current_dict.update(
        layer_config,
      )
      content_losses = np.random.rand(len(base_config["style_paths"]))
      style_losses = np.random.rand(len(base_config["style_paths"]))


      for ith_style, style_path, in enumerate(
        base_config["style_paths"]
      ):
        current_dict = loss_dict = dict(
          style_path=style_path,
          content_loss=content_losses[ith_style],
          style_loss=style_losses[ith_style],
        )
        loss_dict.update(log_dict)

        for k, v in current_dict.items():
          if not isinstance(v, dict):
            df_dict[k].append(v)
    df = pd.DataFrame(df_dict)
    
    figs = get_plotly_fig(
      df,
      title_prefix=f"layer_{layer_idx}",
    )
  
    # for row, fig in enumerate(figs.values()):
    #   fig.show()

    if True:
      import wandb
      run = wandb.init(
        project="test_add_plotly",
        entity="mistake0316",
      )
      
      new_dict = {
        f"plotly_{k}":fig for k, fig in figs.items()
      }
      wandb.log(
        new_dict
      )



  exit()