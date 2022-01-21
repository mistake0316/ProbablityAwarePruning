import torch
from typing import Union
available_modes = ["bias/abs(scale)", "abs(scale)", "bias"]

def ProbablityAwarePruningHook(
    remain_channels:Union[int, float]=1.0,
    mode="bias/abs(scale)",
    importance_scores=None,
    eps = 1e-5, dim=1,
    prune_from_smaller=True,
):  
    std_mean_fun = lambda tensor : torch.std_mean(
        tensor,
        tuple(set(range(tensor.dim()))-set([dim]))
    )

    if importance_scores is not None:
        score_fun = lambda *args, **kwargs : importance_scores
    elif mode == "bias/abs(scale)":
        def score_fun(tensor):
            scale, bias = std_mean_fun(tensor)
            return bias/(torch.abs(scale)+eps)
    elif mode == "bias":
        def score_fun(tensor):
            _, bias = std_mean_fun(tensor)
            return bias
    elif mode == "abs(scale)":
        def score_fun(tensor):
            scale, _ = std_mean_fun(tensor)
            return torch.abs(scale)
    else:
        raise ValueError(f"mode should in {available_modes}, got {mode}")
    
    def inner_hook(cls, input, output):
        LEN = output.shape[dim] # b, >c<, ...
        if isinstance(remain_channels, float):
            channels = int(remain_channels * LEN)
        elif isinstance(remain_channels, int):
            channels = remain_channels
        else:
            raise ValueError(f"type(remain_channels) should in {(float, int)}, got {type(remain_channels)}")
        device = output.device

        mask = torch.zeros(LEN, device=device)
        score = score_fun(output)
        mask[torch.topk(score*(prune_from_smaller*2-1), channels)[1]] = 1
        shape = [1,-1] + [1] * (len(output.shape)-2)
        return mask.reshape(*shape)*output
    
    return inner_hook
    

if __name__ == "__main__":
    out_channels = 5
    in_channels = 3
    tensor = torch.rand(1,in_channels,5,5)
    
    conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3))
    hook = ProbablityAwarePruningHook(
        remain_channels=.6
    )

    print(tensor)
    original_out = conv(tensor)
    print(original_out)
    std, mean = torch.std_mean(original_out, dim=[0,2,3])
    print(std, mean, mean/std)
    conv.register_forward_hook(hook)
    hooked_out = conv(tensor)
    print(hooked_out)