import math
import torch
from torch import cfloat
import torch.nn as nn

from hippo import *
from s4d_kernel import *

## rank = 1
def hippo_LegS(n, k):
    if n > k:
        return -(2*n+1)**0.5 * (2*k+1)**0.5
    if n == k:
        return -(n+1)
    return 0

## Normal Matrix
def hippo_LegS_N_matrix(N, P):
    Ah = hippo_LegS_matrix(N)

    AN = Ah + P @ P.T
    return AN

## Diagonal Matrix
## https://chatgpt.com/share/67a68041-935c-800c-b248-37329bbe4afb
def hippo_LegS_D_matrix(N, P):
    AN = hippo_LegS_N_matrix(N, P)
    eigenvalues, _ = torch.linalg.eig(AN)

    # AD = eigenvalues
    AD = torch.diag(eigenvalues)
    return AD

class Discretize:

    ## S4 paper
    @staticmethod
    def Bilinear(delta, A, B):
        N = A.shape[0]
        I = torch.eye(N)

        A0 = I - delta / 2 * A
        A0 = torch.inverse(A0)
        A1 = I + delta / 2 * A

        Ad = A0 @ A1
        Bd = A0 @ (delta * B)

        return Ad, Bd
    
    ## DSS paper
    @staticmethod
    def ZOH(delta, A, B):
        N = A.shape[0]
        I = torch.eye(N)

        dA = delta * A
        idA = torch.inverse(dA)
        
        ## diagonal matrix inverse
        # idA = 1 / dA

        dB = delta * B

        Ad = torch.exp(dA)
        Bd = idA @ (dA - I) @ dB

        return Ad, Bd

def s4d_kernel(Ad, Bd, C, L):
    ## Ad is a diagonal matrix (NPLR in s4)
    ## Cd = C

    BC = Bd.T * C

    N = Ad.shape[0]
    # print(Ad)
    Ad = torch.diag(Ad)
    Ad = Ad.unsqueeze(1)  # Convert to column vector
    exponent = torch.arange(L)

    print(Ad.shape, exponent.shape)

    ## vandermonde matrix
    VAd = Ad ** exponent

    Kd = BC @ VAd
    Kd = Kd.real
    return Kd

## 1-D input signal x
## N-D latent state A, B, C
## 1-D output signal y
class S4DConv1D(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        ## h'(t) = A h(t) + B x(t)
        ##  y(t) = C h(t)

        ## Before Parameters
        B = torch.arange(N).reshape(N, 1)
        B = (2 * B + 1)**0.5

        P = torch.arange(N).reshape(N, 1)
        P = (P + 1/2)**0.5

        ## parameters
        self.A = nn.Parameter(hippo_LegS_D_matrix(N, P)).to(cfloat) ## N x N
        self.B = nn.Parameter(B).to(cfloat) ## N x 1
        self.C = nn.Parameter(torch.randn(1, N)).to(cfloat) ## 1 x N

        ## scalars
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## step size
        
    ## x: L x 1
    def forward(self, x):
        # h = torch.zeros(self.N)

        L = x.shape[0]

        Ad, Bd = Discretize.ZOH(self.delta, self.A, self.B)
        Kd = s4d_kernel(Ad, Bd, self.C, L)

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