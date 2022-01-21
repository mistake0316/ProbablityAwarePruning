import wandb
import numpy as np
import plotly.express as px


# config = {
#   "style_img_name" : np.random.choice(["A.jpg", "B.jpg", "C.jpg", "D.jpg"]),
#   "layer" : np.random.choice([3,5]),
#   "mode" :  np.random.choice(["a", "b"]),
#   "ratio" : np.random.choice([.3, .5, .7, .9]),
# }

# run = wandb.init(project="my-test-project", entity="mistake0316", config=config)


# for _ in range(10):
#   wandb.log({
#     "content_loss" : {"a":5, "b":3}.get(wandb.config.get("mode"))+np.random.rand()-.5,
#     "style_loss" : {"a":1, "b":3}.get(wandb.config.get("mode"))+np.random.rand()-.5,
#   })

# run.finish()

run = wandb.init(project="3d-plotly", entity="mistake0316")
df = px.data.iris()
wandb.log({
  "Img" : wandb.Image("style_images/cat.jpg")
})
wandb.log({
  "Img" : wandb.Image("style_images/mosaic.jpg")
})
wandb.log({
  "Img2" : wandb.Image("style_images/udnie.jpg")
}, step=0)
