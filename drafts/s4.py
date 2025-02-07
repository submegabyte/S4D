## https://arxiv.org/pdf/2111.00396
## section C.3

## Convolution representation

import math
import torch
from torch import cfloat
import torch.nn as nn

def hippo(i, j):
    if i > j:
        return -(2*i+1)**0.5 * (2*j+1)**0.5
    return -(i+1) if i == j else 0

def hippo_matrix(N):
    A = torch.empty(N, N)

    for i in range(N):
        for j in range(N):
            A[i][j] = hippo(i, j)

    return A

## Diagonal Plus Low Rank Decomposition
## A = Lambda - P Q*
def dplr(A):
    rank = 1
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    Lambda = torch.diag(torch.diag(A)) ## N x N
    P = U[:, :rank] * torch.sqrt(S[:rank])  ## N x 1
    Q = (Vh[:rank, :].conj().T) * torch.sqrt(S[:rank])  ## N x 1

    return Lambda, P, Q

## Cauchy Matrix-Vector Product
def cMV(x, y, Vs):
    ## x - L x 1
    ## y - N x 1
    ## Vs - Suffix Vector N x 1
    ## Compute Mc @ Vs
    ## Mc - Cauchy Matrix L x N
    ## Mc[i][j] = 1 / (x[i] - y[j])
    return ( Vs.reshape(-1) / (x[:, None] - y) ).sum(dim=1) ## L x 1

## L roots of unity
## https://chatgpt.com/share/67a5a858-7ff8-800c-bf10-db349d887632
def omega(L):
    k = torch.arange(L)  # Indices k = 0, 1, ..., L-1
    Omega = torch.exp(2j * torch.pi * k / L)  # Compute roots of unity
    return Omega

## SSM Generating function
def ssmGen(delta, A, B, C, L):
    N = A.shape[0]
    I = torch.eye(N)
    I1 = torch.eye(1)

    ## C tilda star
    ## star = conjugate transpose
    ## Algo 1, step 1
    Ad = torch.inverse(I - delta/2 * A) @ (I + delta/2 * A) ## N x N
    Cts = C.conj().T @ (I - Ad**L) ## 1 X N

    Lambda, P, Q = dplr(A)

    ## R is a diagonal matrix
    xR = lambda z: 2/delta * (1-z)/(1+z) ## scalar
    # R = lambda z: torch.inverse(xR(z) * I - Lambda) ## N x N
    
    ## cauchy matrix
    Omega = omega(L) ## L x 1
    x_Mc = xR(Omega) ## L x 1
    y_Mc = torch.diag(Lambda) ## N x 1


    ## cauchy products 
    ## Algo 1, step 2
    k00 = CRB = cMV(x_Mc, y_Mc, Cts.T * B) ## L x 1
    k01 = CRP = cMV(x_Mc, y_Mc, Cts.T * P) ## L x 1
    k10 = QRP = cMV(x_Mc, y_Mc, Q.conj() * P) ## L x 1
    k11 = QRB = cMV(x_Mc, y_Mc, Q.conj() * B) ## L x 1

    ## using woodbury identity and cauchy products
    ## Algo 1, steps 3 and 4
    # Kc = 2/(1+Omega) * (CRB - CRP / (1 + QRP) * QRB) ## L x 1
    Kc = 2/(1+Omega) * (k00 - k01 / (1 + k10) * k11) ## L x 1
    
    return Kc

## SSM Convolution function using ifft
def ssmConv(delta, A, B, C, L):
    Kc = ssmGen(delta, A, B, C, L) ## L x 1

    ## ifft
    ## Algo 1, step 5
    Kd = torch.fft.ifft(Kc).real.to(torch.float32) ## L x 1
    return Kd

## 1-D input signal
## N-D latent state
## 1-D output signal
class S4Conv1D(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        ## h'(t) = A h(t) + B x(t)
        ## y(t) = C* h(t)

        ## parameters
        self.A = nn.Parameter(hippo_matrix(N)).to(cfloat) ## N x N
        self.B = nn.Parameter(torch.randn(N, 1)).to(cfloat) ## N x 1
        self.C = nn.Parameter(torch.randn(N, 1)).to(cfloat) ## N x 1

        ## scalars
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## step size
        
    ## x: L x 1
    def forward(self, x):
        # h = torch.zeros(self.N)

        L = x.shape[0]

        Kd = ssmConv(self.delta, self.A, self.B, self.C, L)

        # print(Kd.shape, x.shape)
        
        x = Kd @ x
        return x


if __name__ == "__main__":
    L = 20020
    F = 1 ## embedding
    N = 64 ## state

    x = torch.randn(L, F) ## L x 1
    model = S4Conv1D(N, F) ## L x 1

    y = model(x)

    print(y.shape) ## 1 x 1