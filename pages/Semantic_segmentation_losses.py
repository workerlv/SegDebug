from scripts.losses.semantic_pytorch import get_all_pytorch_losses
from scripts.losses.semantic_numpy import get_all_numpy_losses
from scripts.configs.semantic_segm_losses import Const, Config
from scripts.tools.custom_mask import CustomMask
from scripts.utils import mask_utils as m_ut
import streamlit as st
import json


def sidebar():
    st.sidebar.header("Settings")
    Config.gt_figure_width = st.sidebar.number_input(
        "Ground truth figure width", min_value=0, max_value=1000, value=200
    )
    Config.gt_figure_height = st.sidebar.number_input(
        "Ground truth figure height",
        min_value=0,
        max_value=1000,
        value=200,
    )
    Config.pred_figure_width = st.sidebar.number_input(
        "Prediction figure width",
        min_value=0,
        max_value=1000,
        value=200,
    )
    Config.pred_figure_height = st.sidebar.number_input(
        "Prediction figure height",
        min_value=0,
        max_value=1000,
        value=200,
    )


# TODO: need to fix function
def predefined_overlaps():
    option_map = {
        0: "100%",
        1: "75%",
        2: "50%",
        3: "25%",
        4: "0%",
    }

    selection = st.sidebar.pills(
        "Overlaps",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default=[0],
    )

    with open("scripts/configs/box_positions.json", "r") as f:
        data = json.load(f)

    Config.gt_figure_width = data[option_map[selection]]["gt_width"]
    Config.gt_figure_height = data[option_map[selection]]["gt_height"]
    Config.gt_x = data[option_map[selection]]["gt_x"]
    Config.gt_y = data[option_map[selection]]["gt_y"]

    Config.pred_figure_width = data[option_map[selection]]["pred_width"]
    Config.pred_figure_height = data[option_map[selection]]["pred_height"]
    Config.pred_x = data[option_map[selection]]["pred_x"]
    Config.pred_y = data[option_map[selection]]["pred_y"]


def show_losses(gt_mask, pred_mask):
    all_pytorch_losses = get_all_pytorch_losses(
        gt_mask.get_torch_mask(), pred_mask.get_torch_mask()
    )

    all_numpy_losses = get_all_numpy_losses(
        gt_mask.get_numpy_mask(), pred_mask.get_numpy_mask()
    )

    result_df = all_pytorch_losses + all_numpy_losses

    st.dataframe(result_df)
    st.write(f"Overlap: {gt_mask & pred_mask} %")


def main():

    # predefined_overlaps()
    sidebar()

    gt_mask = CustomMask(Const.mask_size, Const.mask_size)
    pred_mask = CustomMask(Const.mask_size, Const.mask_size)

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        gt_horizontal_move = st.slider(
            "GT move horizontaly",
            min_value=0,
            max_value=Const.mask_size - Config.gt_figure_width,
            value=Config.gt_x,
        )

        gt_vertical_move = st.slider(
            "GT move vertically",
            min_value=0,
            max_value=Const.mask_size - Config.gt_figure_height,
            value=Config.gt_y,
        )

        gt_mask.add_mask_with_rectangle(
            Config.gt_figure_width,
            Config.gt_figure_height,
            gt_horizontal_move,
            gt_vertical_move,
        )
        st.image(gt_mask.get_normalized_mask(), caption="Ground truth")

    with col_2:

        pred_horizontal_move = st.slider(
            "Pred move horizontaly",
            min_value=0,
            max_value=Const.mask_size - Config.gt_figure_width,
            value=Config.pred_x,
        )

        pred_vertical_move = st.slider(
            "Pred move vertically",
            min_value=0,
            max_value=Const.mask_size - Config.gt_figure_height,
            value=Config.pred_y,
        )

        pred_mask.add_mask_with_rectangle(
            Config.pred_figure_width,
            Config.pred_figure_height,
            pred_horizontal_move,
            pred_vertical_move,
        )
        st.image(pred_mask.get_normalized_mask(), caption="Predicted")

    with col_3:
        st.write("Color code:")
        st.write("Green: True positives")
        st.write("Red: False negatives")
        st.write("Blue: False positives")

        overlay = m_ut.create_segmentation_overlay(
            gt_mask.get_normalized_mask(), pred_mask.get_normalized_mask()
        )
        st.image(overlay, caption="Overlap")

    if st.button("Show losses"):
        show_losses(gt_mask, pred_mask)


if __name__ == "__main__":
    main()
