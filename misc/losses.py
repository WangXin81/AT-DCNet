import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


def cross_entropy(in_input, target, weight=None, reduction='mean', ignore_index=-100):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if in_input.shape[-1] != target.shape[-1]:
        in_input = F.interpolate(in_input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=in_input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        if self.weight is not None:
            # Apply weights to the loss
            loss = torch.mul(loss, self.weight)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def diceLoss(predicted, target, epsilon=1e-5):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted + target) + epsilon

    dice_loss = 1 - (2 * intersection / union)
    return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = cross_entropy

    def forward(self, predicted, target):
        bce_loss = self.bce_loss(predicted, target)
        dice_loss_val = diceLoss(torch.sigmoid(predicted), target)
        combined_loss = bce_loss + dice_loss_val

        return combined_loss


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(
            self,
            weight: Optional[torch.Tensor] = None,
            ignore_index: Optional[int] = -100,
            reduction: str = "mean",
            pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.pos_weight = 0.5 * y_true
        loss = F.binary_cross_entropy_with_logits(
            y_pred, y_true, self.weight, pos_weight=self.pos_weight, reduction= "mean"
        )

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weight1=None, weight2 = None, gamma = None):
        super(WeightedBCELoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.gamma = gamma

    def forward(self, output, target):
        # 计算二元交叉熵
        loss1 = F.binary_cross_entropy_with_logits(output, target, reduction='none')

        positive_weight = torch.tensor(self.weight1)
        negative_weight = torch.tensor( self.weight2)

        # 应用权重
        weight = torch.where(target == 1, positive_weight, negative_weight)
        prob = torch.exp(-loss1)
        loss = loss1 * (1- prob)**self.gamma * weight
        # loss = weight*loss1

        # 求平均
        loss = torch.mean(loss)

        return loss





