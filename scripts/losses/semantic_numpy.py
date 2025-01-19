from scripts.configs.semantic_segm_losses import Results
from scipy.ndimage import distance_transform_edt
import numpy as np


def calculate_bce_loss(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute Binary Cross-Entropy (BCE) loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).

    Returns:
        float: BCE loss value.
    """
    pred_mask = np.clip(pred_mask, 1e-7, 1 - 1e-7)
    loss = -np.mean(gt_mask * np.log(pred_mask) + (1 - gt_mask) * np.log(1 - pred_mask))
    return loss


def calculate_dice_loss(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute Dice loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).

    Returns:
        float: Dice loss value.
    """
    smooth = 1e-7
    intersection = np.sum(gt_mask * pred_mask)
    dice = (2.0 * intersection + smooth) / (
        np.sum(gt_mask) + np.sum(pred_mask) + smooth
    )
    return 1 - dice


def calculate_iou_loss(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute Intersection over Union (IoU) loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).

    Returns:
        float: IoU loss value.
    """
    smooth = 1e-7
    intersection = np.sum(gt_mask * pred_mask)
    union = np.sum(gt_mask) + np.sum(pred_mask) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou


def calculate_focal_loss(
    gt_mask: np.ndarray, pred_mask: np.ndarray, gamma=2.0, alpha=0.8
):
    """
    Compute Focal loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).
        gamma (float): Focusing parameter.
        alpha (float): Weighting factor for positive class.

    Returns:
        float: Focal loss value.
    """
    pred_mask = np.clip(pred_mask, 1e-7, 1 - 1e-7)  # Avoid log(0)
    loss = -alpha * (1 - pred_mask) ** gamma * gt_mask * np.log(pred_mask) - (
        1 - alpha
    ) * pred_mask**gamma * (1 - gt_mask) * np.log(1 - pred_mask)
    return np.mean(loss)


def calculate_tversky_loss(
    gt_mask: np.ndarray, pred_mask: np.ndarray, alpha=0.5, beta=0.5
):
    """
    Compute Tversky loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).
        alpha (float): Weight for false positives.
        beta (float): Weight for false negatives.

    Returns:
        float: Tversky loss value.
    """
    smooth = 1e-7
    tp = np.sum(gt_mask * pred_mask)
    fp = np.sum((1 - gt_mask) * pred_mask)
    fn = np.sum(gt_mask * (1 - pred_mask))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky


def calculate_boundary_loss(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute Boundary loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).

    Returns:
        float: Boundary loss value.
    """
    boundary_gt = distance_transform_edt(
        1 - gt_mask
    )  # Distance from background to object
    boundary_pred = distance_transform_edt(
        1 - pred_mask.round()
    )  # Thresholded prediction
    loss = np.sum(np.abs(boundary_gt - boundary_pred)) / np.size(gt_mask)
    return loss


def calculate_weighted_bce_loss(
    gt_mask: np.ndarray, pred_mask: np.ndarray, pos_weight=1.0, neg_weight=1.0
):
    """
    Compute Weighted Binary Cross-Entropy (BCE) loss.

    Args:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 or 1).
        pred_mask (np.ndarray): Predicted probabilities (values in [0, 1]).
        pos_weight (float): Weight for positive class.
        neg_weight (float): Weight for negative class.

    Returns:
        float: Weighted BCE loss value.
    """
    pred_mask = np.clip(pred_mask, 1e-7, 1 - 1e-7)  # Avoid log(0)
    loss = -np.mean(
        pos_weight * gt_mask * np.log(pred_mask)
        + neg_weight * (1 - gt_mask) * np.log(1 - pred_mask)
    )
    return loss


def get_all_numpy_losses(target_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:

    results = Results(name="Numpy Losses")
    results.bce_loss = calculate_bce_loss(target_mask, predicted_mask)
    results.dice_loss = calculate_dice_loss(target_mask, predicted_mask)
    results.iou_loss = calculate_iou_loss(target_mask, predicted_mask)
    results.focal_loss = calculate_focal_loss(target_mask, predicted_mask)
    results.tversky_loss = calculate_tversky_loss(target_mask, predicted_mask)
    results.boundary_loss = calculate_boundary_loss(target_mask, predicted_mask)
    results.weighted_bce_loss = calculate_weighted_bce_loss(target_mask, predicted_mask)

    return results
