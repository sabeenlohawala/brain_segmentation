import torch
from torch.nn import functional as F
from torch.nn import NLLLoss

def softmax_focal_loss(
    mask: torch.Tensor,
    probs: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Multi-class version of sigmoid_focal_loss from: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        mask: A tensor of shape (batch_size, nr_of_classes, height, width) containing
              integer class numbers for each pixel.
        probs: A float tensor with the same shape as inputs. Stores the binary
               classification label for each element in inputs
               (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    # Reshape image tensor
    targets = mask.view(-1)  # Shape: (batch_size, 1, height, width) -> (batch_size * height * width,)

    # Reshape softmax output
    p = probs.permute(0, 2, 3, 1).contiguous()  # Move the channel dimension to the end
    p = p.view(-1, probs.shape[1])  # Shape: (batch_size, nr_of_classes, height, width) -> (batch_size * height * width, nr_of_classes)

    loss_fn = NLLLoss(reduction="none")
    ce_loss = loss_fn(p.log(),targets)
    p_t = p[torch.arange(targets.size(0)), targets] # get the probabilities corresponding to the true label
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss, None