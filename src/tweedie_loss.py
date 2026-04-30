"""
Tweedie Loss Implementations
=============================

Standalone implementations of the Tweedie deviance loss function for
NumPy, PyTorch, and TensorFlow. These can be imported directly into
notebooks or production code.

Theory
------
For power parameter p in (1, 2), the Tweedie unit deviance is:

    d(y, mu) = 2 * [ y^(2-p) / ((1-p)(2-p))
                   - y * mu^(1-p) / (1-p)
                   + mu^(2-p) / (2-p) ]

This corresponds to the compound Poisson-Gamma distribution, which
naturally generates zero-inflated, right-skewed positive data.

References
----------
- Jorgensen, B. (1987). Exponential dispersion models. JRSS-B, 49(2), 127-162.
- Delong et al. (2021). Making Tweedie's compound Poisson model more accessible.
  European Actuarial Journal, 11, 185-226.
"""

import numpy as np


# ===========================================================================
# NumPy Implementation
# ===========================================================================

def tweedie_deviance(y_true, y_pred, power=1.5):
    """Compute the mean Tweedie deviance.

    Parameters
    ----------
    y_true : array-like
        Observed (true) values. Must be >= 0.
    y_pred : array-like
        Predicted mean values. Must be > 0.
    power : float, default=1.5
        Tweedie power parameter. Must satisfy 1 < p < 2 for compound
        Poisson-Gamma; p=2 gives Gamma deviance; p=1 gives Poisson deviance.

    Returns
    -------
    float
        Mean Tweedie deviance across all observations.

    Examples
    --------
    >>> tweedie_deviance([0, 0, 100, 200], [10, 5, 80, 250], power=1.5)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 1e-10, None)

    p = power
    if p == 1:
        # Poisson deviance
        dev = 2 * (y_true * np.log(np.clip(y_true, 1e-10, None) / y_pred)
                   - (y_true - y_pred))
    elif p == 2:
        # Gamma deviance
        dev = 2 * (np.log(y_pred / np.clip(y_true, 1e-10, None))
                   + y_true / y_pred - 1)
    else:
        # General Tweedie deviance
        term1 = np.where(
            y_true > 0,
            np.power(y_true, 2 - p) / ((1 - p) * (2 - p)),
            0.0
        )
        term2 = y_true * np.power(y_pred, 1 - p) / (1 - p)
        term3 = np.power(y_pred, 2 - p) / (2 - p)
        dev = 2 * (term1 - term2 + term3)

    return np.mean(dev)


def generate_tweedie_data(n_rows, n_features=10, power=1.5, phi=2.0, seed=42):
    """Generate synthetic Tweedie-distributed data.

    Parameters
    ----------
    n_rows : int
        Number of observations.
    n_features : int, default=10
        Number of predictor features.
    power : float, default=1.5
        Tweedie power parameter (1 < p < 2).
    phi : float, default=2.0
        Dispersion parameter.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_rows, n_features)
        Feature matrix.
    y : ndarray of shape (n_rows,)
        Tweedie-distributed target values.
    """
    rng = np.random.RandomState(seed)
    p = power

    # Generate features
    X = rng.randn(n_rows, n_features).astype(np.float32)

    # Linear predictor
    beta = rng.uniform(-0.3, 0.3, n_features)
    beta[0] = 0.5  # dominant feature
    log_mu = 2.0 + X @ beta
    mu = np.exp(np.clip(log_mu, -5, 10))

    # Compound Poisson-Gamma
    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    beta_g = phi * (p - 1) * mu ** (p - 1)

    N = rng.poisson(lam)
    y = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        if N[i] > 0:
            y[i] = rng.gamma(alpha, beta_g[i], N[i]).sum()

    return X, y


# ===========================================================================
# PyTorch Implementation
# ===========================================================================

def get_pytorch_tweedie_loss(power=1.5):
    """Return a PyTorch Tweedie loss module.

    Parameters
    ----------
    power : float, default=1.5
        Tweedie power parameter.

    Returns
    -------
    nn.Module
        A PyTorch loss module that accepts (log_mu, y_true) tensors.

    Example
    -------
    >>> criterion = get_pytorch_tweedie_loss(p=1.5)
    >>> loss = criterion(model(X), y)
    >>> loss.backward()
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")

    class TweedieLoss(nn.Module):
        """Tweedie deviance loss for PyTorch.

        Accepts raw network output (log-scale) and applies exp() internally
        to ensure positive predictions. This is equivalent to a log-link GLM.

        Parameters
        ----------
        p : float
            Tweedie power parameter, must be in (1, 2).
        """
        def __init__(self, p=1.5):
            super().__init__()
            assert 1.0 < p < 2.0, "Power must be in (1, 2)"
            self.p = p

        def forward(self, log_mu, y_true):
            mu = torch.exp(torch.clamp(log_mu, -10, 10))
            loss = (-y_true * torch.pow(mu + 1e-8, 1 - self.p) / (1 - self.p)
                    + torch.pow(mu + 1e-8, 2 - self.p) / (2 - self.p))
            return torch.mean(loss)

    return TweedieLoss(p=power)


# ===========================================================================
# Convenience: evaluation metrics
# ===========================================================================

def evaluate_predictions(y_true, y_pred, power=1.5, label="Model"):
    """Print a comprehensive evaluation report.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    power : float, default=1.5
        Tweedie power parameter.
    label : str, default="Model"
        Display name.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_safe = np.clip(y_pred, 1e-8, None)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    td = tweedie_deviance(y_true, y_pred_safe, power=power)

    nz = y_true > 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.sum() > 0 else float('inf')
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape, 'Tweedie_Deviance': td}

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  MAE:              {mae:>14,.2f}")
    print(f"  RMSE:             {rmse:>14,.2f}")
    print(f"  MAPE (non-zero):  {mape:>14.1f}%")
    print(f"  WAPE:             {wape:>14.1f}%")
    print(f"  Tweedie Deviance: {td:>14.4f}")
    print(f"  Actual zeros:     {(y_true == 0).sum():>14,d}")
    print(f"  Predicted ~zeros: {(y_pred < 1.0).sum():>14,d}")
    print(f"{'=' * 55}")

    return metrics
