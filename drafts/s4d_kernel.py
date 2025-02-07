## Hadamard Product
## https://chatgpt.com/share/67a684c5-0284-800c-9e30-18db4631fc61


import torch


def s4d_kernel(Ad, Bd, C, L):
    ## Ad is a diagonal matrix (NPLR in s4)
    ## Cd = C

    BC = Bd.T * C

    N = Ad.shape[0]
    Ad = Ad.unsqueeze(1)  # Convert to column vector
    exponent = torch.arange(L)

    ## vandermonde matrix
    VAd = Ad ** exponent

    Kd = BC @ VAd
    return Kd
