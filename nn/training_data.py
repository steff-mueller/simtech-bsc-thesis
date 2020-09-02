import torch

from nn.linearsymplectic import SymplecticScaling

"""
    Scales every pair (q,p) so that q and p have approximately same magnitude.
"""
def scale_training_data(y):
    dim = y.shape[1]
    assert dim % 2 == 0, 'Dimension must be even.'
    dim_half = int(dim/2)

    y_q = y[:,0:dim_half]
    y_p = y[:,dim_half:dim]
    a = torch.sqrt(torch.abs((y_p+1e-8)/(y_q+1e-8))) # prevent division by zero

    # Note that `a` has shape (n,dim), because we want to scale
    # each training pair. SymplecticScaling normally expects (dim,).
    # But shape (n,dim) is still working because of the way 
    # the SymplecticScaling layer is implemented.
    scaler = SymplecticScaling(dim, a)
    scaler.a.requires_grad = False
    return scaler(y), scaler