import pandas as pd

import re

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.pyplot import imread


def draw_emoji(
    ax, key, position, zoom=0.2, code=None, frameon=False, xycoords="data", xybox=None
):
    img = load_emoji(key, code=code)
    if img is None:
        return

    if xybox is None:
        xybox = (0, 0)

    imagebox = OffsetImage(img, zoom=zoom, interpolation="hanning")

    ab = AnnotationBbox(
        imagebox,
        position,
        xycoords=xycoords,
        xybox=xybox,
        boxcoords="offset points",
        frameon=frameon,
    )

    ax.add_artist(ab)


PREFIX = re.compile("^u|^U0+")


def remove_prefix(key):
    return PREFIX.sub("", key.encode("unicode-escape")[1:].decode("ascii"))


# @memoize
def load_emoji(key, code=None, path=None):
    if path is None:
        path = "../twemoji/assets/72x72"
    if code is None:
        img_code = "-".join(map(remove_prefix, key))

        try:
            img = imread("{}/{}.png".format(path, img_code))
        except FileNotFoundError:
            print(key, remove_prefix(key), img_code, "not found")
            return None
    else:
        img = read_png("{}/{}.png".format(path, code))
    return img


from emoji.unicode_codes import EMOJI_DATA
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# fix - this emoji has multiple codes
# https://www.iemoji.com/view/emoji/40/symbols/red-heart
EMOJI_DATA["❤️"]["en"] = ":red_heart:"

import matplotlib.pyplot as plt
import seaborn as sns


def replace_emoji_in_ticks(ax, tick_values, scale=1, placeholder="      "):
    TICK_POS = -0.6

    ticks = []
    for i, term in enumerate(tick_values):
        if ":" in term:
            parts = term.split(":", maxsplit=1)
        else:
            parts = ["", term]

        if parts[1] not in EMOJI_DATA:
            if parts[0] == "":
                ticks.append(parts[1])
            else:
                ticks.append(term)
            continue

        try:
            emoji_str = parts[1].replace(u'\ufe0f', '')
            imagebox = OffsetImage(
                load_emoji(emoji_str), zoom=scale * 0.125, interpolation="hanning"
            )
            imagebox.image.axes = ax

            ab = AnnotationBbox(
                imagebox,
                (0, i + 0.125 * scale * 2.75),
                xybox=(-4 * scale, -0.1),
                xycoords=("axes points", "data"),
                boxcoords="offset points",
                box_alignment=(1.5, 1.1),
                bboxprops={"edgecolor": "none"},
            )

            ax.add_artist(ab)
            if parts[0] != "":
                ticks.append(parts[0] + ":" + placeholder)
            else:
                ticks.append("")
        except TypeError:
            ticks.append(EMOJI_DATA[parts[1]]["en"])

        

    # ax.get_yaxis().set_ticklabels([t.replace('term:', '') for t in features['new_label']], fontsize=10)
    ax.get_yaxis().set_ticklabels(ticks, fontsize=10)


def feature_importance_plot(
    features,
    figsize=(7, 12),
    replace_emoji=True,
    k=50,
    classification_type="Stance",
    color="#FFB7C5",
):

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    features = features.assign(
        new_label=lambda x: x["new_label"].str.replace("term:", "")
    )

    (
        features.set_index("new_label")["xgb.relevance"].plot(
            kind="barh", ax=ax, legend=False, color=color, width=0.9
        )
    )

    TICK_POS = -0.6

    replace_emoji_in_ticks(ax, features["new_label"])

    box_inset = plt.axes([0.475, 0.5, 0.4, 0.2], facecolor="#efefef")

    # box_inset = inset_axes(ax, width='40%', height='20%', loc='center right')
    feature_freqs = features["type"].value_counts().sort_values(ascending=True)

    feature_freqs.plot(kind="barh", width=0.85, ax=box_inset, color=color)
    box_inset.set_ylabel("")
    box_inset.set_xlabel("Frecuency")
    box_inset.set_title("Frequency per Feature Type")

    box_inset = plt.axes([0.475, 0.2, 0.4, 0.2], facecolor="#efefef")

    # box_inset = inset_axes(ax, width='40%', height='20%', loc='center right')
    sns.barplot(
        y="type",
        x="xgb.relevance",
        data=features,
        # kind='bar',
        color=color,
        ax=box_inset,
        order=reversed(feature_freqs.index),
        alpha=1.0,
    )
    box_inset.set_ylabel("")
    box_inset.set_xlabel("Mean Importance")
    box_inset.set_title("Importance per Feature Type")

    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(
        f"{classification_type} Classifier: Top-{len(features)} Important Features"
    )
    sns.despine(ax=ax, left=True)
    # sns.despine(ax=box_inset)
    return fig, ax


def proportion_chart(
    ax,
    proportions,
    labels,
    colors,
    avoid_undisclosed=True,
    min_fraction=5,
    title=None,
    xlabel=None,
):
    bar_labels = proportions.cumsum(axis=1) - (
        proportions / 2
    )  # .sort_index(ascending=False)

    (
        proportions.plot(
            kind="barh",
            stacked=True,
            width=0.9,
            edgecolor="none",
            linewidth=0.4,
            color=colors,
            ax=ax,
        )
    )

    ax.set_yticklabels(proportions.index.values)

    i = 0
    for idx, row in bar_labels.iterrows():
        for key in proportions.columns:
            if avoid_undisclosed and key.startswith("undisclosed"):
                continue
            print(proportions.loc[idx, key])
            if proportions.loc[idx, key] < min_fraction:
                continue
            ax.annotate(
                f"{proportions.loc[idx, key]:.2f}%",
                (row[key], i),
                ha="center",
                va="center",
                fontsize=12,
            )
        i += 1

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim([-1, 101])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        ax.legend(
            labels.values(),
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
        )
    else:
        ax.legend(loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
    sns.despine(ax=ax, left=True)
