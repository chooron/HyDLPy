import torch
import torch.nn as nn


class NSELoss(nn.Module):
    """
    Nash-Sutcliffe Efficiency loss.

    Returns 1 - NSE, so that minimizing the loss maximizes the NSE.
    A value of 1 corresponds to a perfect match, and a value of 0 corresponds to the model being no better than the mean.
    """

    def __init__(self, eps: float = 1e-6):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate NSE loss.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.

        Returns
        -------
        torch.Tensor
            The NSE loss value.
        """
        numerator = torch.sum(torch.pow(y_pred - y_true, 2))
        denominator = torch.sum(torch.pow(y_true - torch.mean(y_true),  2))

        nse = 1 - (numerator / (denominator + self.eps))

        return 1 - nse

class LogNSELoss(nn.Module):
    """
    Nash-Sutcliffe Efficiency loss.

    Returns 1 - NSE, so that minimizing the loss maximizes the NSE.
    A value of 1 corresponds to a perfect match, and a value of 0 corresponds to the model being no better than the mean.
    """

    def __init__(self, eps: float = 1e-6):
        super(LogNSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate NSE loss.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.

        Returns
        -------
        torch.Tensor
            The NSE loss value.
        """
        numerator = torch.sum(torch.pow(y_pred - y_true, 2))
        denominator = torch.sum(torch.pow(y_true - torch.mean(y_true),  2))

        return torch.log((numerator / (denominator + self.eps))+1)


class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency loss.

    Returns 1 - KGE, so that minimizing the loss maximizes the KGE.
    KGE is a composite metric that combines correlation, bias, and variability.
    """

    def __init__(self, eps: float = 1e-6):
        super(KGELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate KGE loss.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.

        Returns
        -------
        torch.Tensor
            The KGE loss value.
        """
        # Pearson correlation coefficient
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)

        vx = y_pred - mean_pred
        vy = y_true - mean_true

        r_num = torch.sum(vx * vy)
        r_den = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        r = r_num / (r_den + self.eps)

        # Beta (bias term)
        beta = mean_pred / (mean_true + self.eps)

        # Gamma (variability term)
        std_pred = torch.std(y_pred)
        std_true = torch.std(y_true)
        gamma = (std_pred / (mean_pred + self.eps)) / (std_true / (mean_true + self.eps) + self.eps)

        # KGE calculation
        kge = 1 - torch.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

        return 1 - kge
