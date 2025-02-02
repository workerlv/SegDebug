import numpy as np
import torch
import cv2


class Mask:
    def __init__(self, segmentation_mask):
        self.mask = segmentation_mask
        self.check_if_mask_binary()

    def check_if_mask_binary(self):
        if np.max(self.mask) > 1:
            self.mask = self.mask / 255
            self.mask = self.mask.astype(np.uint8)

    def get_normalized_mask(self):
        return self.mask * 255

    def get_torch_mask(self):
        return torch.from_numpy(self.mask.astype(np.float32))

    def get_numpy_mask(self):
        return self.mask

    def __and__(self, other):
        """
        Calculate the percentage of overlap between two CustomMask objects.

        Args:
            other (CustomMask): Another CustomMask object.

        Returns:
            float: Percentage of overlap between the two masks.
        """
        if not isinstance(other, Mask):
            raise ValueError("Operand must be an instance of CustomMask.")

        # Ensure the masks are of the same shape
        if self.mask.shape != other.mask.shape:
            raise ValueError("Masks must have the same shape.")

        # Calculate intersection and union
        intersection = np.sum(self.mask * other.mask)
        union = np.sum(np.clip(self.mask + other.mask, 0, 1))

        if union == 0:
            return 0.0  # No union means no overlap is possible

        # Calculate overlap percentage
        overlap_percentage = (intersection / union) * 100
        return int(round(overlap_percentage, 0))
