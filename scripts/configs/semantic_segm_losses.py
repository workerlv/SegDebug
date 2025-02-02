from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


class Const:
    mask_size = 1000


class Config:
    # Use example data
    use_example_data: bool = True

    # Ground truth
    gt_figure_width: int = 0
    gt_figure_height: int = 0
    gt_x: int = 100
    gt_y: int = 100

    # Prediction
    pred_figure_width: int = 0
    pred_figure_height: int = 0
    pred_x: int = 100
    pred_y: int = 100


@dataclass
class Results:
    name: str
    bce_loss: Optional[float] = field(default=None)
    dice_loss: Optional[float] = field(default=None)
    iou_loss: Optional[float] = field(default=None)
    focal_loss: Optional[float] = field(default=None)
    tversky_loss: Optional[float] = field(default=None)
    boundary_loss: Optional[float] = field(default=None)
    weighted_bce_loss: Optional[float] = field(default=None)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Results attributes to a pandas DataFrame with name as column header."""
        data = {
            "bce_loss": [self.bce_loss],
            "dice_loss": [self.dice_loss],
            "iou_loss": [self.iou_loss],
            "focal_loss": [self.focal_loss],
            "tversky_loss": [self.tversky_loss],
            "boundary_loss": [self.boundary_loss],
            "weighted_bce_loss": [self.weighted_bce_loss],
        }
        df = pd.DataFrame(data).T
        df.columns = [self.name]
        return df

    def __add__(self, other: "Results") -> pd.DataFrame:
        """Concatenate DataFrames from two Results instances using their names as column headers."""
        if not isinstance(other, Results):
            return NotImplemented

        df1 = self.to_dataframe()
        df2 = other.to_dataframe()

        return pd.concat([df1, df2], axis=1)
