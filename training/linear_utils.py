"""
Closed-form OLS regression for AQUA-KV cross-layer predictor training.

Ported from aquakv/linear_utils.py — dropped unused reduced_rank_regression.
"""
from typing import Optional

import torch
from tqdm import trange


@torch.no_grad()
def fit_linear_regression(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    reg_rate: float = 0,
    fit_intercept: bool = True,
    compute_device: Optional[torch.device] = None,
    compute_dtype: Optional[torch.dtype] = None,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
):
    """
    Multivariate linear regression with exact solution.

    :param X: input, shape [nsamples, in_features]
    :param Y: target, shape [nsamples, num_targets]
    :param reg_rate: regularization proportional to mean square input
    :param fit_intercept: if True, learn bias term
    :param compute_device: device for computing the pseudoinverse
    :param compute_dtype: dtype for accumulation (memory-efficient)
    :param chunk_size: process this many rows at a time for large datasets
    :returns: (W, b) such that Y ~ (X @ W.T + b)
    """
    assert X.ndim == Y.ndim == 2 and X.shape[0] == Y.shape[0]
    (nsamples, in_features), (_, out_features), source_device = X.shape, Y.shape, Y.device
    orig_dtype = X.dtype

    if chunk_size is None:
        X = X.to(compute_dtype)
        if fit_intercept:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        CXX = (X.T @ X).to(compute_device)
    else:
        CXX = torch.zeros(
            in_features + 1, in_features + 1, device=compute_device, dtype=compute_dtype or X.dtype
        )
        for chunk_start in trange(
            0, nsamples, chunk_size, desc="fit_linear_regression::CXX", leave=False, disable=not verbose
        ):
            xb = X[chunk_start : chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True
            )
            if fit_intercept:
                xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=xb.device)], dim=1)
            CXX = torch.addmm(CXX, xb.T, xb, out=CXX)
            del xb

    if reg_rate > 0:
        ix = torch.arange(len(CXX), device=compute_device or source_device)
        CXX[ix, ix] += reg_rate * abs(torch.diag(CXX)).mean()
        del ix

    CXX_pinv = torch.pinverse(CXX)
    del CXX

    if chunk_size is None:
        CXY = (X.T @ Y).to(compute_device)
        del X, Y
    else:
        CXY = torch.zeros(
            in_features + 1, out_features, device=compute_device, dtype=compute_dtype or X.dtype
        )
        for chunk_start in trange(
            0, nsamples, chunk_size, desc="fit_linear_regression::CXY", leave=False, disable=not verbose
        ):
            xb, yb = [
                tensor[chunk_start : chunk_start + chunk_size].to(
                    device=compute_device, dtype=compute_dtype, non_blocking=True
                )
                for tensor in (X, Y)
            ]
            if fit_intercept:
                xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=xb.device)], dim=1)
            CXY = torch.addmm(CXY, xb.T, yb, out=CXY)
            del xb, yb
        del X, Y

    W = (CXX_pinv @ CXY).T.to(source_device, dtype=orig_dtype)

    if fit_intercept:
        W, bias = W[:, :-1], W[:, -1]
    else:
        bias = None

    return W, bias
