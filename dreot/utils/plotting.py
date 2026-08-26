import numpy as np
import pandas as pd


def highlight_minmax(df, split_XY=False):
    """
    Highlight max (green), 2nd max (light green), and min (red) per column.

    Parameters
    ----------
    df : pd.DataFrame
    split_XY : bool, default False
        If True, apply highlighting separately within the "X" and "Y" levels
        of a ("W", "dataset") MultiIndex.
    """
    def _color_col(col):
        styles = [""] * len(col)
        max_val = col.max()
        min_val = col.min()
        below_max = col[col < max_val]
        second_max_val = below_max.max() if not below_max.empty else None
        for i, v in enumerate(col):
            if v == max_val:
                styles[i] = "background-color: #2ecc71"
            elif v == second_max_val:
                styles[i] = "background-color: #a9dfbf"
            elif v == min_val:
                styles[i] = "background-color: lightcoral"
        return styles

    if not split_XY:
        return df.style.apply(_color_col, axis=0)

    def _color_full(df):
        style_df = pd.DataFrame("", index=df.index, columns=df.columns)
        for ds in ["X", "Y"]:
            sub_df = df.xs(ds, level="dataset")
            for col in sub_df.columns:
                col_styles = _color_col(sub_df[col])
                for i, w in enumerate(sub_df.index):
                    style_df.loc[(w, ds), col] = col_styles[i]
        return style_df

    return df.style.apply(_color_full, axis=None)
