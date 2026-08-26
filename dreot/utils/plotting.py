import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_composition_piecharts(adata1, adata2, column, labels=("Dataset 1", "Dataset 2"), figsize=(18, 6)):
    """
    Side-by-side pie charts showing cluster composition of two AnnData objects.
    The same cluster uses the same color in both charts.

    Parameters
    ----------
    adata1, adata2 : AnnData
    column : str
        obs column to use (e.g. "cluster").
    labels : tuple of str
        Titles for the two pie charts.
    figsize : tuple of float, default (18, 6)
        Figure size passed to plt.subplots.
    """
    counts1 = adata1.obs[column].value_counts()
    counts2 = adata2.obs[column].value_counts()

    all_clusters = sorted(set(counts1.index) | set(counts2.index))
    colors = cm.tab20(np.linspace(0, 1, len(all_clusters)))
    color_map = dict(zip(all_clusters, colors))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, counts, title in zip(axes, [counts1, counts2], labels):
        ax.pie(
            counts.values,
            labels=counts.index,
            colors=[color_map[c] for c in counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"{title}\n(n={counts.sum()})")
    plt.tight_layout()
    plt.show()


def plot_pct_mknn(
    df,
    figsize=(8, 5),
    xlabel="k",
    ylabel="pct_mknn score",
    title="Label-concordant mutual kNN score across methods and k values",
):
    """
    Line plot of pct_mknn scores for each W matrix across k values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of evaluate_pct_mknn. Rows = W names, columns = "k={k}".
    figsize : tuple of float, default (8, 5)
        Figure size passed to plt.subplots.
    xlabel : str, default "k"
        Label for the x-axis.
    ylabel : str, default "pct_mknn score"
        Label for the y-axis.
    title : str, default "Label-concordant mutual kNN score across methods and k values"
        Title for the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name in df.index:
        ax.plot(df.columns, df.loc[name], marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def highlight_minmax(df, split_XY=False):
    """
    Highlight max (green), 2nd max (light green), and min (red) per column.

    Parameters
    ----------
    df : pd.DataFrame
    split_XY : bool, default False
        If True, apply highlighting separately within the "X" and "Y" levels
        of a ("W", "dataset") MultiIndex (as produced by
        evaluate_pct_mknn_per_Wdict). Replaces the separate highlight_minmax_XY
        function — call as highlight_minmax(df, split_XY=True).
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
