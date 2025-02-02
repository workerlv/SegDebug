import numpy as np
import cv2


def create_segmentation_overlay(gt_mask, pred_mask):
    """
    Creates a 3-channel image to visualize segmentation results.

    Parameters:
        gt_mask (np.ndarray): Ground truth mask (binary: 0 and 1).
        pred_mask (np.ndarray): Predicted mask (binary: 0 and 1).

    Returns:
        np.ndarray: 3-channel image with the following color scheme:
            - Black (0, 0, 0): Background.
            - Green (0, 255, 0): Correct predictions (overlapping masks).
            - Red (255, 0, 0): Missed ground truth (false negatives).
            - Blue (0, 0, 255): False positives.
    """
    # Ensure binary values (0 or 1)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # Create empty 3-channel image
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

    # Green: Correct predictions (intersection of ground truth and prediction)
    correct_predictions = gt_mask & pred_mask
    overlay[correct_predictions == 1] = [0, 255, 0]

    # Red: Missed ground truth (ground truth present but no prediction)
    missed_ground_truth = gt_mask & ~pred_mask
    overlay[missed_ground_truth == 1] = [255, 0, 0]

    # Blue: False positives (prediction present but no ground truth)
    false_positives = ~gt_mask & pred_mask
    overlay[false_positives == 1] = [0, 0, 255]

    return overlay


def create_geen_overlay(image, mask, alpha=0.7):
    """
    Applies a green overlay on the original image using the given segmentation mask.

    Parameters:
        image (numpy.ndarray): Original image (BGR format).
        mask (numpy.ndarray): Segmentation mask (grayscale or binary format, same width/height as image).
        alpha (float): Transparency of the overlay (0.0 to 1.0).

    Returns:
        numpy.ndarray: Image with green overlay applied.
    """
    # Ensure mask is binary
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Create an all-green image
    green_overlay = np.zeros_like(image, dtype=np.uint8)
    green_overlay[:, :] = (0, 255, 0)

    # Apply the mask to the green overlay
    green_overlay = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)

    # Blend the original image and the green overlay
    output = cv2.addWeighted(image, 1 - alpha, green_overlay, alpha, 0)

    return output
