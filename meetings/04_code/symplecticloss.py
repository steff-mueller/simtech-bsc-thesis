import torch

def compute_jacobian(outputs: torch.Tensor, inputs: torch.Tensor, device=None):
    k = outputs.size()[0]
    n = inputs.size()[0]
    
    res = torch.zeros(k, n).to(device)

    for i in range(0, n):
        grad_outputs = torch.zeros(n).to(device)
        grad_outputs[i] = 1.
        grad = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True)[0]
        res[:,i:i+1] = grad.reshape(2,1)

    return res.t() # transpose because autograd.grad calculates J.t() * v

def symplectic_loss(y_target, model, x, device=None):
    mse_d = 0.
    mse_s = 0.  

    J = torch.tensor([[0., 1.], [-1., 0.]]).to(device)

    n = x.size()[0]
    for i in range(0,n):
        x_i = x[i]
        y_target_i = y_target[i]
        y_pred_i = model(x_i)
        jacobian = compute_jacobian(y_pred_i, x_i, device)

        mse_d += torch.sum((y_pred_i - y_target_i) ** 2)

        symp = jacobian.t().mm(J).mm(jacobian) - J
        mse_s += torch.sum(symp ** 2)
        
    w = 0.1
    return 1/n*(mse_d + w*mse_s)