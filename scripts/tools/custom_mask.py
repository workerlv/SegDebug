from enum import Enum
import numpy as np
import torch
import cv2


class CustomMask:
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

    def add_mask_with_circle(self, radius: int, x_cord: int, y_cord: int):
        self.mask = cv2.circle(self.mask, (x_cord, y_cord), radius, 1, -1)

    def add_mask_with_rectangle(
        self, width: int, height: int, x_cord: int, y_cord: int
    ):
        self.mask[y_cord : y_cord + height, x_cord : x_cord + width] = 1

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
        if not isinstance(other, CustomMask):
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
