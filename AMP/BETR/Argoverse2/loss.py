import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PytorchLogMeanDisplacementError(nn.Module):
    """
    Compute the log mean displacement error between the ground truth and the prediction.
    """
    def __init__(self):
        super(PytorchLogMeanDisplacementError, self).__init__()

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Ground truth tensor of shape (batch_size, time, 2).
            y_pred (torch.Tensor): Prediction tensor of shape (batch_size, modes, time, 2).

        Returns:
            torch.Tensor: Log mean displacement error.
        """
        y = torch.unsqueeze(y, 1)  # add modes dimension

        # Compute error
        error = torch.sum((y - y_pred) ** 2, dim=-1)  # (batch_size, modes, time)
        
        # Compute log mean displacement error
        log_mde = -torch.logsumexp(error, dim=-1, keepdim=True)
        return torch.mean(log_mde)


class PytorchNegMultiLogLikelihoodBatch(nn.Module):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    """
    def __init__(self):
        super(PytorchNegMultiLogLikelihoodBatch, self).__init__()

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Ground truth tensor of shape (batch_size, time, 2).
            y_pred (torch.Tensor): Prediction tensor of shape (batch_size, modes, time, 2).
            confidences (torch.Tensor): Tensor of shape (batch_size, modes) with confidence for each mode.

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        y = torch.unsqueeze(y, 1)  # add modes dimension

        # Compute error
        error = torch.sum((y - y_pred) ** 2, dim=-1)  # (batch_size, modes, time)
        
        # Compute negative log-likelihood
        log_confidences = F.log_softmax(confidences, dim=1) - 0.5 * torch.sum(error, dim=-1)
        nll = -torch.logsumexp(log_confidences, dim=-1, keepdim=True)
        return torch.mean(nll)


def mean_displacement_error(y: torch.Tensor, y_pred: torch.Tensor) -> list:
    """
    Compute the mean displacement error between the ground truth and the prediction.

    Args:
        y (torch.Tensor): Ground truth tensor of shape (batch_size, time, 2).
        y_pred (torch.Tensor): Prediction tensor of shape (batch_size, modes, time, 2).

    Returns:
        list: List of mean displacement errors for each mode.
    """
    errors = []
    for i in range(y_pred.shape[1]):
        error = torch.sqrt(torch.sum((y - y_pred[:, i]) ** 2, dim=-1))
        errors.append(error.mean().item())
    return errors


def final_displacement_error(y: torch.Tensor, y_pred: torch.Tensor) -> list:
    """
    Compute the final displacement error between the ground truth and the prediction.

    Args:
        y (torch.Tensor): Ground truth tensor of shape (batch_size, time, 2).
        y_pred (torch.Tensor): Prediction tensor of shape (batch_size, modes, time, 2).

    Returns:
        list: List of final displacement errors for each mode.
    """
    errors = []
    for i in range(y_pred.shape[1]):
        error = torch.sqrt(torch.sum((y[:, -1] - y_pred[:, i, -1]) ** 2, dim=-1))
        errors.append(error.mean().item())
    return errors


def missrate(y_true: torch.Tensor, y_pred: torch.Tensor, heading: torch.Tensor, comb: str = 'avg') -> torch.Tensor:
    """
    Compute the miss rate between the ground truth and the prediction.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (batch_size, time, 2).
        y_pred (torch.Tensor): Prediction tensor of shape (batch_size, modes, time, 2).
        heading (torch.Tensor): Heading angles tensor of shape (batch_size,).
        comb (str): Combination method ('avg' or 'min').

    Returns:
        torch.Tensor: Miss rate tensor of shape (batch_size, 80).
    """
    R = torch.zeros((len(heading), 2, 2)).to(heading.device)
    R[:, 0, 0] = torch.cos(heading)
    R[:, 0, 1] = -torch.sin(heading)
    R[:, 1, 0] = torch.sin(heading)
    R[:, 1, 1] = torch.cos(heading)

    MR = torch.zeros((len(y_pred), 80), device=y_pred.device) if comb == 'avg' else torch.ones((len(y_pred), 80), device=y_pred.device)
    
    lat = [1, 1.8, 3]
    lon = [2, 3.6, 6]
    samples = [slice(30), slice(30, 50), slice(50, 80)]

    for i in range(y_pred.shape[1]):
        err = y_true - y_pred[:, i]
        err = torch.einsum('bij,bjk->bik', err, R)

        for j in range(len(lat)):
            if comb == 'min':
                MR[:, samples[j]] = torch.where(
                    (torch.abs(err[:, samples[j], 0]) > lat[j]) | (torch.abs(err[:, samples[j], 1]) > lon[j]),
                    torch.zeros_like(MR[:, samples[j]]), MR[:, samples[j]]
                )
            elif comb == 'avg':
                MR[:, samples[j]] += torch.where(
                    (torch.abs(err[:, samples[j], 0]) > lat[j]) | (torch.abs(err[:, samples[j], 1]) > lon[j]),
                    torch.ones_like(MR[:, samples[j]]), torch.zeros_like(MR[:, samples[j]])
                )

    if comb == 'avg':
        MR = MR / y_pred.shape[1]
    return MR
