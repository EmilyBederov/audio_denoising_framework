# models/cleanunet2/util.py
import torch
import torch.nn as nn

def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)

def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        # Total number of parameters (trainable and non-trainable)
        total_params = sum(p.numel() for p in net.parameters())
        # Number of trainable parameters
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        print("{} - Total Parameters: {:.6f}M; Trainable Parameters: {:.6f}M".format(
            net.__class__.__name__, total_params / 1e6, trainable_params / 1e6), flush=True)
        
        if keyword is not None:
            # Parameters associated with the keyword
            keyword_params = [p for name, p in net.named_parameters() if keyword in name]
            keyword_total_params = sum(p.numel() for p in keyword_params)
            keyword_trainable_params = sum(p.numel() for p in keyword_params if p.requires_grad)
            
            print("'{0}' - Total Parameters: {1:.6f}M; Trainable Parameters: {2:.6f}M".format(
                keyword, keyword_total_params / 1e6, keyword_trainable_params / 1e6), flush=True)
