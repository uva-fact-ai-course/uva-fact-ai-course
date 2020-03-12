import torch


def bias_regularization_term(embd, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term.
    """
    if norm:
        embd = embd / embd.norm(dim=1).view(-1, 1)

    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size(0)):
        idxs = D[idx].view(-1)
        u = embd[idxs[0],:]
        v = embd[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)

    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S.pow(2)

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)

    # Get first k components to for gender subspace
    B = V[:, :k_idx.item()+1]
    loss = torch.matmul(embd[N], B).norm().pow(2)

    return lmbda * loss
