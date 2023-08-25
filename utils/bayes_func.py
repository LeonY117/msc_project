import torch

from torch import Tensor
from typing import Optional, Tuple


def bayes_forward(
    net,
    X: torch.tensor,
    k: int,
    mode: Optional[str] = "all",
    buffer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Performs k forward passes with stochastic regularisation

    Args
    -----------
    net: nn.Module
    X  : torch.tensor (c x W x H), a single input image
    k  : int, indicating number of repeated forwards passes
    mode: str, indicating which bayesian mode to use
    buffer (optional): torch.tensor (k x c x W x H) buffer

    Returns
    -----------
    y_softmax            : torch.tensor (c x W x H)
    y_pred               : torch.tensor (W x H)
    y_pred_std_per_class : torch.tensor (c x W x H)
    y_pred_std_avg       : torch.tensor (W x H)
    """
    if buffer is None:
        buffer = X.unsqueeze(0).repeat(k, 1, 1, 1)
    else:
        for i in range(k):
            # write image to buffer
            buffer[i] = X

    with torch.no_grad():
        net.eval()
        net.set_bayes_mode(True, mode)
        y_logits = net(buffer) # (k x c x W x H)

    # Average the softmax (note that the resultant vectors are not normalised)
    # y_logits = y_pred_raw.mean(dim=0)  # (k x c x W x H)
    y_softmax = y_logits.softmax(dim=1)  # (k x c x W x H)
    y_softmax_avg = y_softmax.mean(dim=0)  # (c x W x H)
    # Take max prob as prediction
    y_pred = torch.argmax(y_softmax_avg, dim=0).to(torch.int)  # (W x H)
    # Per class uncertainty
    y_pred_std_per_class = y_softmax.var(dim=0)  # (c x W x H)
    # Average uncertainty over classes
    y_pred_std_avg = y_pred_std_per_class.mean(dim=0)  # (W x H)

    return y_softmax_avg, y_pred, y_pred_std_per_class, y_pred_std_avg


def bayes_eval(
    net,
    X: Tensor,
    k: int,
    mode: Optional[str] = "all",
    buffer: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs k forward passes with dropout layers, returns prediction

    Args
    -----------
    net: nn.Module
    X  : torch.tensor (c x W x H), a single input image
    k  : int, indicating number of repeated forwards passes
    mode: str, indicating which bayesian mode to use
    buffer (optional): torch.tensor(k x c x W x H) buffer

    Returns
    -----------
    y_pred               : torch.tensor (W x H)
    """

    with torch.no_grad():
        net.eval()
        if k == 0:
            net.set_bayes_mode(False, "all")
            buffer = X
        elif k > 0:
            net.set_bayes_mode(True, mode)
            # write image to buffer
            if buffer is None:
                buffer = X.unsqueeze(0).repeat(k, 1, 1, 1)
            else:
                for i in range(k):
                    buffer[i] = X

    y_pred_raw = net(buffer)
    y_pred = torch.argmax(y_pred_raw.mean(dim=0), dim=0).to(torch.int8)  # (W x H)

    return y_pred
