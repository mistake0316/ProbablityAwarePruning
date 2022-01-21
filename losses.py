import torch
import torchvision
from torchvision import transforms as T
from PIL import Image

# Following magenta original loss
#   https://github.com/magenta/magenta/blob/main/magenta/models/arbitrary_image_stylization/arbitrary_image_stylization_train.py#L29-L31
# DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1}'
# DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
#                          ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')

device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision import models
import torch

imagenet_preprocess = torchvision.transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225],
)

class VGG(torch.nn.Module):
  def __init__(
    self,
    preprocesser=imagenet_preprocess,
    requires_grad=False,
    model_name="vgg16",
  ):
    super(VGG, self).__init__()
    assert "vgg" in model_name
    
    self.preprocesser = preprocesser
    vgg_features = getattr(models, model_name)(pretrained=True).features
    
    features = self.features = torch.nn.ModuleDict()
    
    block=1
    idx=1
    for layer in vgg_features:
      if isinstance(layer, torch.nn.Conv2d):
        features[f"conv{block}_{idx}"] = layer
      elif isinstance(layer, torch.nn.ReLU):
        features[f"relu{block}_{idx}"] = torch.nn.ReLU(inplace=False)
        idx += 1
      elif isinstance(layer, torch.nn.MaxPool2d):
        features[f"pool{block}"] = layer
        idx = 1
        block += 1
      else:
        raise ValueError(f"{type(layer)} is not allow now.")
    
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad=False
  
  @property
  def layer_names(self):
    return list(self.features.keys())

  def forward(self, input, outkeys=None):
    outdict = {}
    x = input
    if self.preprocesser:
      x = self.preprocesser(x)
    for name, layer in self.features.items():
      x = layer(x)
      if outkeys and name in outkeys:
        outdict[name] = x
    if not outkeys:
      return x
    else:
      return outdict


vgg16 = vgg = VGG()

# DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1}'
def content_loss(
  tensor1,
  tensor2,
  weight_layer_pairs=[
    (1, "relu3_3")
  ]
):
  weights, layers = zip(*weight_layer_pairs)
  
  out_dict_1 = vgg16(tensor1, layers)
  out_dict_2 = vgg16(tensor2, layers)

  loss = 0
  for w, l in zip(weights, layers):
    loss += w*torch.mean((out_dict_1[l]-out_dict_2[l])**2)
  return loss


def gram_matrix(y):
    b,c,h,w = y.size()
    gram = torch.einsum("bihw,bjhw->bij",y,y)
    return gram/(c*h*w)
  
# DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
#                          ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')
def style_loss(
  tensor1,
  tensor2,
  weight_layer_pairs=[
    (5e-4, "relu1_2"),
    (5e-4, "relu2_2"),
    (5e-4, "relu3_3"),
    (5e-4, "relu4_3"),
  ],
):
  weights, layers = zip(*weight_layer_pairs)
  
  out_dict_1 = vgg16(tensor1, layers)
  out_dict_2 = vgg16(tensor2, layers)

  loss = 0
  for w, l in zip(weights, layers):
    gram_1 = gram_matrix(out_dict_1[l])
    gram_2 = gram_matrix(out_dict_2[l])
    loss += w*torch.mean((gram_1-gram_2)**2)
  return loss

if __name__ == "__main__":
  print(vgg.layer_names)
  tensors = [
    T.Resize(384)(T.ToTensor()(Image.open("doge.jpg")).unsqueeze(0))
  ]
  tensors.append(
    tensors[0]+torch.rand_like(tensors[0])*.1
  )
  
  with torch.no_grad():
    print(f"content loss : {content_loss(*tensors).item()}")
    print(f"style loss : {style_loss(*tensors).item()}")