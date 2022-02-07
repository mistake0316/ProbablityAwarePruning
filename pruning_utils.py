import torch
from typing import Union
available_modes = ["bias/abs(scale)", "abs(scale)", "bias", "from_first_layers"]

def ProbablityAwarePruningHook( # for convolution
    remain_channels:Union[int, float]=1.0,
    mode="bias/abs(scale)",
    importance_scores=None,
    eps = 1e-5, dim=1,
    prune_from_smaller=True,
):  
    std_mean_fun = lambda tensor : torch.std_mean(
        tensor,
        tuple(set(range(tensor.dim()))-set([0, dim])),
        keepdim=True,
    )

    if importance_scores is not None:
        score_fun = lambda *args, **kwargs : importance_scores * (prune_from_smaller*2-1)
    elif mode in available_modes:
        def score_fun(tensor, mode=mode):
            scale, bias = std_mean_fun(tensor)
            original_score = {
                "bias/abs(scale)":bias/(torch.abs(scale)+eps),
                "bias":bias,
                "abs(scale)":torch.abs(scale),
                "from_first_layers":torch.arange(tensor.shape[1]),
            }.get(mode, None)
            if original_score is None:
                raise NotImplementedError()

            return original_score * (prune_from_smaller*2-1)
    else:
        raise ValueError(f"mode should in {available_modes}, got {mode}")
    
    def inner_hook(cls, input, output):
        assert isinstance(cls, torch.nn.ReLU)
        input = input[0]
        LEN = input.shape[dim] # b, >c<, ...
        B = input.shape[0] # >b<, c, ...
        if isinstance(remain_channels, float):
            channels = int(remain_channels * LEN)
        elif isinstance(remain_channels, int):
            channels = remain_channels
        else:
            raise ValueError(f"type(remain_channels) should in {(float, int)}, got {type(remain_channels)}")
        device = output.device

        mask = torch.zeros((B, LEN), device=device)
        score = score_fun(input)
        if score.dim() == 1:
            score = score.unsqueeze(0).repeat(B,1)
        indices = torch.topk(score, k=channels, dim=dim)[1]
        batch_indices = (
            torch.arange(indices.shape[0],device=device)
                .unsqueeze(1)
                .repeat((1,indices.shape[1]))
        )
        mask[(batch_indices.reshape(-1), indices.reshape(-1))] = 1
        shape = [B,-1] + [1] * (len(output.shape)-2)
        return mask.view(*shape) * output
    
    return inner_hook
    

if __name__ == "__main__":
    out_channels = 5
    in_channels = 3
    tensor = torch.rand(2,in_channels,5,5)
    
    conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3))
    hook = ProbablityAwarePruningHook(
        remain_channels=.6
    )

    print(tensor)
    original_out = conv(tensor)
    print(original_out)
    std, mean = torch.std_mean(original_out, dim=[2,3])
    print(std, mean, mean/std)
    conv.register_forward_hook(hook)
    hooked_out = conv(tensor)
    print(hooked_out)