import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(self, input, target, smooth=1):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        input = input.view(-1)
        target = target.view(-1)
        intersection = (input * target).sum()
        dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
        return 1 - dice


# --- Binary Cross Entropy Loss ---
class BCELoss(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(BCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid

    def forward(self, input, target):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        return F.binary_cross_entropy(input.view(-1), target.view(-1))


# --- Dice + BCE Loss ---
class DiceBCELoss(nn.Module):
    def __init__(self, use_sigmoid=True, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.smooth = smooth

    def forward(self, input, target):
        if self.use_sigmoid:
            input = torch.sigmoid(input)
        input = input.view(-1)
        target = target.view(-1)

        # Dice Loss
        intersection = (input * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                input.sum() + target.sum() + self.smooth
        )

        # BCE Loss
        bce = F.binary_cross_entropy(input, target)

        return bce + dice_loss


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = input.clamp(self.eps, 1 - self.eps)
        target = target.float()

        logit = input.view(-1)
        target = target.view(-1)

        bce = F.binary_cross_entropy(logit, target, reduction='none')
        focal_weight = (1 - logit) ** self.gamma
        focal_loss = (bce * focal_weight).mean()

        return focal_loss


# --- Dice + Focal Loss ---
class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1, eps=1e-7):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = input.clamp(self.eps, 1 - self.eps)
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        # Dice Loss
        intersection = (input_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                input_flat.sum() + target_flat.sum() + self.smooth
        )

        # Focal Loss
        bce = F.binary_cross_entropy(input_flat, target_flat, reduction='none')
        focal_weight = (1 - input_flat) ** self.gamma
        focal_loss = (bce * focal_weight).mean()

        return focal_loss + dice_loss


# --- Label Smoothing Cross-Entropy Loss (NEW!) ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements label smoothing regularization as described in Eq.11 and Eq.13 of your paper.
    Converts hard labels to soft ones to improve generalization.
    """
    def __init__(self, alpha=0.1, num_classes=2):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        targets = targets.view(-1)

        # Create smoothed labels
        device = inputs.device
        true_dist = torch.zeros(batch_size, self.num_classes, device=device)
        true_dist.fill_(self.alpha / (self.num_classes - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), 1 - self.alpha)

        # Compute cross entropy with smoothed labels
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(log_probs * true_dist).sum(dim=1).mean()

        return loss