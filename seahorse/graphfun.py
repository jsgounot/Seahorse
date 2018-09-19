# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-05-16 13:53:18
# @Last modified by:   jsgounot
# @Last Modified time: 2018-05-16 14:25:48

# http://patorjk.com/software/taag/#p=display&v=3&f=Calvin%20S&t=barplot
# Calvin S

from math import pi, degrees

import pylab as plt
import pandas as pd

from matplotlib.collections import PathCollection

import seaborn as sns

from seahorse import graph_utils, constants

"""
╔═╗┌─┐┬  ┌─┐┬─┐┌─┐┌┬┐  ╦═╗┌─┐┌─┐┌─┐┬  ┌─┐┌┬┐
║  │ ││  │ │├┬┘├┤  ││  ╠╦╝├┤ │ ┬├─┘│  │ │ │ 
╚═╝└─┘┴─┘└─┘┴└─└─┘─┴┘  ╩╚═└─┘└─┘┴  ┴─┘└─┘ ┴
 """

def colored_regplot(col1, col2, data, ax, cbar=None, cbar_label=None, hue=None, hue_color=None, 
 	kws_scatter={}, kwg_corr={}, ** kwargs) :
    
    graph_utils.remove_non_number(data, (col1, col2))

    if cbar is not None :
        # http://stackoverflow.com/questions/13943217/how-to-add-colorbars-to-scatterplots-created-like-this
        cmap = sns.cubehelix_palette(light=.9, as_cmap=True)
        graph_utils.remove_non_number(data, (cbar, ))
        third_variable = data[cbar]
        skws = {"c" : third_variable, "cmap" : cmap, "color" : None}
        skws.update(kws_scatter)
        sns.regplot(col1, col2, data=data, ax=ax, scatter_kws=skws, ** kwargs)

        outpathc = [child for child in ax.get_children()
        if isinstance(child, PathCollection)][0]
        color_bar = plt.colorbar(mappable=outpathc, ax=ax)
        cbar_label = cbar if cbar_label is None else cbar_label
        color_bar.set_label(cbar_label)

    elif hue is not None :
        if "color" in kwargs : kwargs.pop("color")
        for sname, sdf in data.groupby(hue) :
            color = hue_color[sname] if hue_color and sname in hue_color else None
            sns.regplot(col1, col2, data=sdf, ax=ax, label=sname, color=color, ** kwargs)
        
        ax.legend()

    else :
        sns.regplot(col1, col2, data=data, ax=ax, ** kwargs)

    return data[[col1, col2]].corr(**kwg_corr).iat[0,1]

"""
╔═╗┌─┐┌┬┐┌─┐┬  ┌─┐┌┬┐
║  ├─┤ │ ├─┘│  │ │ │ 
╚═╝┴ ┴ ┴ ┴  ┴─┘└─┘ ┴
"""

def custom_plt(x, y, data, ax, hue=None, legend=True, colors=None, ** kwargs) :
    if hue :
        for idx, col in enumerate(data[hue].unique()) :
            subdf = data[data[hue] == col]
            color = colors[idx] if colors else None
            custom_plt(x, y, subdf, ax, color=color, ** kwargs)
    
        if legend : ax.legend(data[hue].unique())
        return

    else :
        ax.plot(data[x], data[y], ** kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

def plot_cat(df, y, xhue, ax, yhue=None, color=None, legend=True, fill=False, ** kwargs) :
    # special plt function for cat plot
    if yhue :
        colors = {}
        for idx, group in enumerate(df.groupby(yhue)) :
            name, sdf = group
            color = sns.color_palette()[idx]
            plot_cat(sdf, y, xhue, ax, None, color, legend=legend, ** kwargs)
            colors[name] = color

        if legend :
            elements = [mlines.Line2D([], [], color=color, label=label) for label, color in colors.items()]
            ax.legend(handles=elements)

    else :
        for name, sdf in df.groupby(xhue) :
            custom_plt("gposi", y, sdf, ax, color=color, ** kwargs)
            if fill : ax.fill_between(sdf["gposi"], 0, sdf[y], alpha=.5)

def cat_plot(x, y, xhue, data, ax, yhue=None, funsort=None, legend=True, tick_rot=0, background=False, start_zero=False,
             xhue_size={}, bgcolor=None, fill=False, ** kwargs) :

    df = data.copy()
    # remove rows for which we only have one row for a xhue (I don't remember why)
    # to_remove = [name for name, sdf in df.groupby(xhue) if len(sdf.drop_duplicates(x)) == 1]
    # df = df[~df[xhue].isin(to_remove)]
    df["cat_show"] = True

    if xhue_size :
        dflen = pd.DataFrame([{x : xhue_v, y : 0, xhue : xhue_e, "cat_show" : False}
        for xhue_e, xhue_v in xhue_size.items()])
        df = pd.concat([df, dflen])

    if start_zero :
        dfzero = pd.DataFrame([{x : 0, y : 0, xhue : xhue_e, "cat_show" : False}
        for xhue_e in df[xhue].unique()])
        df = pd.concat([df, dfzero])

    # Add graph posi
    if funsort is None : funsort = lambda x : (len(str(x)), x)
    distinctxhue = sorted(df[xhue].unique(), key=funsort)
    df[xhue] = pd.Categorical(df[xhue], distinctxhue)
    df = df.sort_values([xhue, x])

    # Calculation of the graphical position of each point
    # additionner contains the value to add for each xhue values based on the maximum value of x
    # for each precedent xhue. This maximum value represents the value which must be add for all the point of
    # the next chromosome. This value is then shifted to the bottom and the first value is set to 0 (fillna)
    # this can work since the df is correctly sorted from before and because groupby reads the dataframe
    # from the top to the bottom (so the order is correctly respected by groupby)
    
    additionner = df.groupby(xhue)[x].max().cumsum().shift(1).fillna(0).to_dict()
    df["gposi"] = df.apply(lambda row : additionner[row[xhue]] + row[x], axis=1)

    # plot data
    plot_cat(df[df["cat_show"] == True], y, xhue, ax, yhue, None, legend, fill, ** kwargs)

    # add ticks and labels
    ticksinfo = [(name, (sdf["gposi"].min() + sdf["gposi"].max()) / 2.) for name, sdf in df.groupby(xhue)]
    ax.set_xticks([tickinfo[1] for tickinfo in ticksinfo])
    ax.set_xticklabels([tickinfo[0] for tickinfo in ticksinfo], rotation=tick_rot)
    
    # add labels
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # add background for pair idx
    if background :
        bgcolor = "0.5" if bgcolor is None else bgcolor
        extra = df["gposi"].max() * 0.01
        extra = 0
        for idx, group in enumerate(df.groupby(xhue)) :
            if idx % 2 == 0 : continue
            name, sdf = group
            ax.axvspan(sdf["gposi"].min() - extra, sdf["gposi"].max() + extra,
            facecolor=bgcolor, alpha=0.5)

    ax.set_xlim((df["gposi"].min(), df["gposi"].max()))

    return df.groupby(xhue).min()["gposi"].to_dict()

"""
┌┐ ┌─┐┬─┐┌─┐┬  ┌─┐┌┬┐
├┴┐├─┤├┬┘├─┘│  │ │ │ 
└─┘┴ ┴┴└─┴  ┴─┘└─┘ ┴ 
"""

def barplot_twinx(left, right, data, ax, colors=None, width=.8, border_size=.5, ylabels=(None, None)) :  
    left = data[[left]] if isinstance(left, str) else data[left] 
    right = data[[right]] if isinstance(right, str) else data[right] 

    left_cn = len(left.columns)
    right_cn = len(right.columns)

    colors = sns.color_palette() if colors is None else colors
    column_width = width / (left_cn + right_cn)

    df1pos = .5 + (right_cn / left_cn) * .5
    df2pos = .5 - (left_cn / right_cn) * .5

    left.plot(kind='bar', width=column_width * left_cn, 
        color=colors[:left_cn], position=df1pos, ax=ax)

    ax2 = ax.twinx()
    right.plot(kind='bar', width=column_width * right_cn, 
        color=colors[left_cn:right_cn+left_cn], position=df2pos, ax=ax2)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax2.grid(b=False, axis='both')
    ax.set_xlim((-border_size, len(data) - border_size))

    if ylabels[0] : ax.set_ylabel(ylabels[0])
    if ylabels[1] : ax2.set_ylabel(ylabels[1])

    return (ax, ax2)

def stacked_barplot(x, y, hue, data, ax, prop=False, sort_values=False, stack_order=None, * args, ** kwargs) :
    ndf = pd.pivot_table(data, values=y, index=x, columns=hue)
    if prop : ndf = ndf.apply(lambda x : x / x.sum(), axis=1)
    ndf = ndf.fillna(0)

    if stack_order :
        ndf = ndf[stack_order]

    if sort_values and not prop : 
        if isinstance(sort_values, Iterable) :
            ndf = ndf.T[list(sort_values)].T
        else :
            ndf["sum"] = ndf.sum(axis=1)
            ndf = ndf.sort_values("sum", ascending=False).drop("sum", 1)

    ndf.plot.bar(stacked=True, ax=ax, * args, ** kwargs)

def dist_barplot(column, bin_size, data, ax, filler=None, colors=None, * args, ** kwargs) :
    colors = colors or constants.DEFAULT_COLOR
    bins = (data[column] // bin_size) * bin_size
    bins = bins.value_counts()

    if filler is not None :
        bins = bins.to_dict()
        bins = {value : bins.get(value, 0) for value in filler}
        bins = pd.Series(bins)

    bins = bins.rename("count").to_frame().reset_index()
    sns.barplot(x="index", y="count", data=bins, ax=ax, color=colors, * args, ** kwargs)


"""
╔═╗┬┬─┐┌─┐┬ ┬┬  ┌─┐┬─┐  ┌┐ ┌─┐┬─┐┌─┐┬  ┌─┐┌┬┐
║  │├┬┘│  │ ││  ├─┤├┬┘  ├┴┐├─┤├┬┘├─┘│  │ │ │ 
╚═╝┴┴└─└─┘└─┘┴─┘┴ ┴┴└─  └─┘┴ ┴┴└─┴  ┴─┘└─┘ ┴ 
"""

def cbar_offset_sep(sep_width) :
    return (pi * 2) * sep_width

def cbar_slice(df)  :
    return (2 * pi) / len(df)

def cbar_slice_size(df, cvalue) :
    return cbar_slice(df) / df[cvalue].max()

def cbar_transform_df(df, corr_values, cvalue, slice, slice_size, offset_sep) :
    if corr_values :
        df["theta"] = (df[cvalue] * slice_size) - offset_sep
    else :
        df["theta"] = slice - offset_sep

    df["radii"] = [slice * i for i in range(len(df))]
    df["label_pos"] = df["radii"].apply(degrees)
    return df

def circular_barplot(cname, cvalue, data, ax=None, palette=None, sep_width=0.02, bottom=0, 
    corr_values=True, grid_below=True) :
    
    df = data

    slice = cbar_slice(df)
    slice_size = cbar_slice_size(df, cvalue)
    offset_sep = cbar_offset_sep(sep_width)

    df = df.groupby(cname)[cvalue].mean().to_frame()
    df = cbar_transform_df(df, corr_values, cvalue, slice, slice_size, offset_sep)
    
    palette = palette or sns.color_palette()
    bars = ax.bar(df["radii"], df[cvalue], width=df["theta"], bottom=bottom, color=palette)

    min_rlab_pos = df[df[cvalue] == df[cvalue].min()]["label_pos"].tolist()[0]

    ax.set_xticklabels([])
    ax.set_thetagrids(df["label_pos"].tolist(), df.index.tolist())
    ax.set_rlabel_position(min_rlab_pos)
    ax.set_aspect('equal')

    # to populated the axis
    ax.set_yticklabels(ax.get_yticks())
    labels = [label.get_text() for label in ax.get_yticklabels()]

    ax.set_yticklabels([label if float(label) > df[cvalue].min() else ""
    for label in labels])

    if grid_below : ax.set_axisbelow(True)