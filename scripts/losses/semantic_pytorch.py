from scripts.configs.semantic_segm_losses import Results
import torch.nn as nn
import torch


def calculate_bce_loss(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor
) -> float:
    """
    Calculate Binary Cross Entropy (BCE) loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].

    Returns:
        torch.Tensor: BCE loss value.
    """
    # Ensure both tensors have the same shape
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    bce_loss_fn = nn.BCELoss()

    # Reshape tensors to (Batch x -1) for loss calculation
    target_flat = target_mask.view(-1)
    predicted_flat = predicted_mask.view(-1)
    bce_loss = bce_loss_fn.forward(predicted_flat, target_flat)
    return round(bce_loss.item(), 5)


def calculate_dice_loss(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor, smooth: float = 1e-6
) -> float:
    """
    Calculate Dice loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].
        smooth (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss value.
    """
    # Ensure both tensors have the same shape
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    # Reshape tensors to (Batch x -1) for loss calculation
    target_flat = target_mask.view(target_mask.size(0), -1)
    predicted_flat = predicted_mask.view(predicted_mask.size(0), -1)

    # Calculate intersection and union
    intersection = torch.sum(target_flat * predicted_flat, dim=1)
    union = torch.sum(target_flat, dim=1) + torch.sum(predicted_flat, dim=1)

    # Dice coefficient and loss
    dice_coefficient = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coefficient.mean()

    return round(dice_loss.item(), 5)


def calculate_iou_loss(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate IoU (Jaccard) loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].
        smooth (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: IoU loss value.
    """
    # Ensure both tensors have the same shape
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    # Reshape tensors to (Batch x -1) for loss calculation
    target_flat = target_mask.view(target_mask.size(0), -1)
    predicted_flat = predicted_mask.view(predicted_mask.size(0), -1)

    # Calculate intersection and union
    intersection = torch.sum(target_flat * predicted_flat, dim=1)
    union = (
        torch.sum(target_flat, dim=1) + torch.sum(predicted_flat, dim=1) - intersection
    )

    # IoU score and loss
    iou_score = (intersection + smooth) / (union + smooth)
    iou_loss = 1 - iou_score.mean()

    return round(iou_loss.item(), 5)


def calculate_focal_loss(
    target_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    alpha: float = 0.8,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Calculate Focal loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].
        alpha (float): Weighting factor for class imbalance (default 0.8).
        gamma (float): Focusing parameter to down-weight easy examples (default 2.0).

    Returns:
        torch.Tensor: Focal loss value.
    """
    # Ensure both tensors have the same shape
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    # Reshape tensors to (Batch x -1) for loss calculation
    target_flat = target_mask.view(-1)
    predicted_flat = predicted_mask.view(-1)

    # Avoid log(0) by clamping predicted values
    predicted_flat = torch.clamp(predicted_flat, min=1e-6, max=1 - 1e-6)

    # Calculate Binary Cross-Entropy
    bce_loss = -(
        target_flat * torch.log(predicted_flat)
        + (1 - target_flat) * torch.log(1 - predicted_flat)
    )

    # Apply Focal Loss formula
    focal_loss = (
        alpha * (1 - predicted_flat) ** gamma * target_flat * bce_loss
        + (1 - alpha) * predicted_flat**gamma * (1 - target_flat) * bce_loss
    )

    return round(focal_loss.mean().item(), 5)


def calculate_tversky_loss(
    target_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate Tversky loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].
        alpha (float): Weight for false positives (default 0.5).
        beta (float): Weight for false negatives (default 0.5).
        smooth (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Tversky loss value.
    """
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    target_flat = target_mask.view(target_mask.size(0), -1)
    predicted_flat = predicted_mask.view(predicted_mask.size(0), -1)

    true_positives = torch.sum(target_flat * predicted_flat, dim=1)
    false_positives = torch.sum(predicted_flat * (1 - target_flat), dim=1)
    false_negatives = torch.sum((1 - predicted_flat) * target_flat, dim=1)

    tversky_index = (true_positives + smooth) / (
        true_positives + alpha * false_positives + beta * false_negatives + smooth
    )
    tversky_loss = 1 - tversky_index.mean()

    return round(tversky_loss.item(), 5)


def calculate_boundary_loss(
    target_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    boundary_weight: float = 1.0,
) -> torch.Tensor:
    """
    Calculate boundary loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].
        boundary_weight (float): Weight for boundary pixels.

    Returns:
        torch.Tensor: Boundary loss value.
    """
    if target_mask.shape != predicted_mask.shape:
        raise ValueError("Target and predicted masks must have the same shape.")

    from scipy.ndimage import distance_transform_edt

    # Compute boundary distance map
    boundary_map = torch.from_numpy(
        distance_transform_edt(1 - target_mask.cpu().numpy())
    ).to(target_mask.device)
    boundary_map = boundary_map / (boundary_map.max() + 1e-6)

    # Loss weighted by boundary map
    weighted_loss = (
        nn.BCELoss(reduction="none")(predicted_mask, target_mask) * boundary_map
    )
    return round(weighted_loss.mean().item(), 5)


def calculate_weighted_bce_loss(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Weighted BCE loss for binary masks.

    Args:
        target_mask (torch.Tensor): Ground truth binary mask (Batch x H x W) with values in {0, 1}.
        predicted_mask (torch.Tensor): Predicted binary mask (Batch x H x W) with probabilities in [0, 1].

    Returns:
        torch.Tensor: Weighted BCE loss value.
    """
    # Compute class weights
    foreground_weight = 1.0 / (
        target_mask.sum() + 1e-6
    )  # Inverse of foreground frequency
    background_weight = 1.0 / (
        (1 - target_mask).sum() + 1e-6
    )  # Inverse of background frequency

    # Assign weights to each pixel
    weights = torch.where(target_mask == 1, foreground_weight, background_weight)

    # Define BCE loss function with reduction='none'
    bce_loss_fn = nn.BCELoss(reduction="none")

    # Compute unweighted BCE loss
    bce_loss = bce_loss_fn(predicted_mask, target_mask)

    # Apply weights
    weighted_loss = bce_loss * weights

    # Return mean loss
    return round(weighted_loss.mean().item(), 5)


def get_all_pytorch_losses(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor
) -> dict:

    results = Results(name="PyTorch Losses")
    results.bce_loss = calculate_bce_loss(target_mask, predicted_mask)
    results.dice_loss = calculate_dice_loss(target_mask, predicted_mask)
    results.iou_loss = calculate_iou_loss(target_mask, predicted_mask)
    results.focal_loss = calculate_focal_loss(target_mask, predicted_mask)
    results.tversky_loss = calculate_tversky_loss(target_mask, predicted_mask)
    results.boundary_loss = calculate_boundary_loss(target_mask, predicted_mask)
    results.weighted_bce_loss = calculate_weighted_bce_loss(target_mask, predicted_mask)

    return results
