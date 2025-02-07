## Vandermonde Matrix
## https://chatgpt.com/c/67a684ee-ff18-800c-bc34-b646ced36879

import torch

def vandermonde(x, N, increasing=False):
    """
    Creates a Vandermonde matrix using PyTorch.
    
    Args:
        x (torch.Tensor): 1D tensor of input values.
        N (int): Number of columns in the output matrix.
        increasing (bool): If True, powers increase from left to right.
                           If False, powers decrease from left to right.
    
    Returns:
        torch.Tensor: Vandermonde matrix.
    """
    x = x.unsqueeze(1)  # Convert to column vector
    exponent = torch.arange(N, dtype=x.dtype, device=x.device)
    
    if not increasing:
        exponent = exponent.flip(0)  # Reverse for decreasing order

    return x ** exponent  # Broadcasting applies exponentiation

# Example usage:
x = torch.tensor([1, 2, 3], dtype=torch.float32)  # Input values
N = 4  # Number of columns
V = vandermonde(x, N, increasing=False)
print(V)