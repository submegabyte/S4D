## Section 3.1

import torch

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
        I = torch.eye(N)
        
        dA = delta * A
        idA = torch.inverse(dA)

        dB = delta * B

        Ad = torch.exp(dA)
        Bd = idA @ (dA - I) @ dB

        return Ad, Bd