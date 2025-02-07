## https://arxiv.org/pdf/2111.00396
## section C.1

import torch

######################

## S4

import torch.nn as nn

def hippo_LagT(n, k):
    if n < k:
        return 0
    if n == k:
        return -0.5
    return -1

## rank = 1
def hippo_LegS(n, k):
    if n > k:
        return -(2*n+1)**0.5 * (2*k+1)**0.5
    if n == k:
        return -(n+1)
    return 0

def hippo_LegT(n, k):
    if n >= k:
        return -1
    return -(-1)**(n+k)

N = 8

def hippo_LegS_matrix(N):
    A = torch.empty(N, N)

    for i in range(N):
        for j in range(N):
            A[i][j] = hippo_LegS(i, j)

    return A

## NPLR Matrix
## A = hippo_LegS_matrix(N)

######################

## S4D

B = torch.arange(N).reshape(N, 1)
B = (2 * B + 1)**0.5

P = torch.arange(N).reshape(N, 1)
P = (P + 1/2)**0.5

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

    AD = torch.diag(eigenvalues)
    return AD

## Diagonal Matrix
A = hippo_LegS_D_matrix(N, P)