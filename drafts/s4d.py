import math
import torch
from torch import cfloat
import torch.nn as nn

from hippo import *
from s4d_kernel import *

# def hippo(i, j):
#     if i > j:
#         return -(2*i+1)**0.5 * (2*j+1)**0.5
#     return -(i+1) if i == j else 0

# def hippo_matrix(N):
#     A = torch.empty(N, N)

#     for i in range(N):
#         for j in range(N):
#             A[i][j] = hippo(i, j)

#     return A

# def s4d_kernel(delta, A, B, C, L):
#     vandermonde = torch.exp(torch.arange(L)[:,None] * delta * A)
#     K = vandermonde * B * C * (torch.exp(delta * A) - 1) / A ## N x L ???
#     K = torch.sum(K, dim=-1)
#     return K

## 1-D input signal
## N-D latent state
## 1-D output signal
class S4DConv1D(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        ## h'(t) = A h(t) + B x(t)
        ##  y(t) = C h(t)

        ## parameters
        self.A = nn.Parameter(hippo_matrix(N)).to(cfloat) ## N x N
        self.B = nn.Parameter(torch.randn(N, 1)).to(cfloat) ## N x 1
        self.C = nn.Parameter(torch.randn(1, N)).to(cfloat) ## 1 x N

        ## scalars
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## step size
        
    ## x: L x 1
    def forward(self, x):
        # h = torch.zeros(self.N)

        L = x.shape[0]

        Kd = s4d_kernel(self.delta, self.A, self.B, self.C, L)

        # print(Kd.shape, x.shape)
        
        x = Kd @ x
        return x


if __name__ == "__main__":
    L = 20020
    F = 1 ## embedding
    N = 64 ## state

    x = torch.randn(L, F) ## L x 1
    model = S4DConv1D(N, F) ## L x 1

    y = model(x)

    print(y.shape) ## 1 x 1