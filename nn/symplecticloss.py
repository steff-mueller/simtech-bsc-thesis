import torch

# TODO only works for 2D systems for now
def symplectic_mse_loss(x, y1, retain_graph=False, device=None):
    N = x.size(0)

    grad_outputs = torch.zeros(y1.size())
    grad_outputs[:,0] = 1
    grad_q = torch.autograd.grad(y1, x, grad_outputs=grad_outputs, retain_graph=True)[0]

    grad_outputs[:,0] = 0
    grad_outputs[:,1] = 1
    grad_p = torch.autograd.grad(y1, x, grad_outputs=grad_outputs, retain_graph=retain_graph)[0]

    grad_q = grad_q.unsqueeze(1)
    grad_p = grad_p.unsqueeze(1)

    grad = torch.cat([grad_q, grad_p], dim=1)

    J = torch.tensor([
        [0., 1.],
        [-1., 0.]
    ])

    symplectic_error = grad.transpose(1,2).matmul(J.matmul(grad)) - J
    return 1/N*torch.norm(symplectic_error)