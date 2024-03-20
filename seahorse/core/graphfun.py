# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-05-16 13:53:18
# @Last modified by:   jsgounot
# @Last Modified time: 2024-03-20 14:13:06

# http://patorjk.com/software/taag/#p=display&v=3&f=Calvin%20S&t=barplot
# Calvin S

import itertools
from math import pi, degrees

import pylab as plt
import pandas as pd
import numpy as np

import matplotlib.lines as mlines
from matplotlib.collections import PathCollection
from matplotlib.patches import Circle

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

from seahorse.core import graph_utils, constants
from seahorse.core.gwrap import sns
from seahorse.custom.venn import venn_df, venn_dic


def plot(x, y, data, ax, hue=None, palette=None, color=None, fill=0, fbeetween=None, ** kwargs) :
    # Similar to pandas.plot (using behind) function but with the correct argument
    # Meaning that you can use x, y and hue, and you don't have to transform the df before

    if hue : data = pd.pivot_table(data, index=x, values=y, columns=hue).fillna(fill)
    else : data = data.set_index(x)[y]

    if color and palette is None:
        palette = sns.set_palette(sns.color_palette([color]))

    palette = palette or sns.color_palette()
    try : colors = [palette[name] for name in data.columns] if hue else palette[0]
    except KeyError : colors = [palette[idx] for idx in range(len(data.columns))] if hue else palette[0]
    except TypeError : colors = [palette[idx] for idx in range(len(data.columns))] if hue else palette[0]
  
    kwargs["color"] = colors

    r = data.plot(ax=ax, ** kwargs)


def colored_regplot(col1, col2, data, ax, cbar=None, cbar_label=None, hue=None, hue_color=None, 
 	kws_scatter={}, kwg_corr={}, ** kwargs) :
    
    graph_utils.remove_non_number(data, (col1, col2))

    if cbar is not None :
        # http://stackoverflow.com/questions/13943217/how-to-add-colorbars-to-scatterplots-created-like-this
        cmap = sns.cubehelix_palette(light=.9, as_cmap=True)
        data = graph_utils.remove_non_number(data, (cbar, ))
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

    correlation = data[[col1, col2]].corr(** kwg_corr)
    return correlation

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

def plot_cat(df, y, xhue, ax, yhue=None, color=None, legend=True, fill=False, palette=None, ** kwargs) :
    # special plt function for cat plot

    if yhue :
        colors = {}
        for idx, group in enumerate(df.groupby(yhue)) :
            name, sdf = group
            icolor = color or (palette[idx] if palette else sns.color_palette()[idx])
            plot_cat(sdf, y, xhue, ax, None, icolor, legend=legend, fill=fill, ** kwargs)
            colors[name] = icolor

        if legend :
            elements = [mlines.Line2D([], [], color=icolor, label=label) for label, icolor in colors.items()]
            ax.legend(handles=elements)

    else :
        idx_name = 0
        for name, sdf in df.groupby(xhue) :
                        
            if palette : 
                color = palette[idx_name]
                idx_name += 1

            custom_plt("gposi", y, sdf, ax, color=color, ** kwargs)
            if fill : ax.fill_between(sdf["gposi"], 0, sdf[y], alpha=.5, color=color)

def cat_plot(x, y, xhue, data, ax, yhue=None, funsort=None, legend=True, tick_rot=0, background=False, start_zero=False,
             xhue_size={}, bgcolor=None, fill=False, palette=None, ** kwargs) :

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
    color = kwargs.get("color", None) if palette is None else None
    kwargs = {key : value for key, value in kwargs.items() if key not in ("color",)}    
    plot_cat(df[df["cat_show"] == True], y, xhue, ax, yhue, color, legend, fill, palette, ** kwargs)

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


def kdeplothue(data, ax, hue, value, palette=None, ** kwargs) :

    for huen, sdf in data.groupby(hue) :
        sdf = sdf[value]
        sns.kdeplot(data=sdf, ax=ax, ** kwargs)

    ax.legend()

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

def stacked_barplot(x, y, hue, data, ax, prop=False, sort_values=False, 
        stack_order=None, palette=None, ignore=[], horizontal=False, 
        * args, ** kwargs) :
    
    ndf = pd.pivot_table(data, values=y, index=x, columns=hue)
    if prop : ndf = ndf.apply(lambda x : x / x.sum(), axis=1)
    ndf = ndf.fillna(0)

    if stack_order :
        ndf = ndf[stack_order]

    if horizontal :
        ndf = ndf.iloc[::-1]

    if ignore :
        used = [column for column in ndf.columns if column not in ignore]
        ndf = ndf[used]

    if sort_values and not prop : 
        if isinstance(sort_values, Iterable) :
            ndf = ndf.T[list(sort_values)].T
        else :
            ndf["sum"] = ndf.sum(axis=1)
            ndf = ndf.sort_values("sum", ascending=False).drop("sum", 1)

    if palette and isinstance(palette, dict) :
        colors = [palette[column] for column in ndf.columns]
    elif palette and isinstance(palette, list) :
        colors = palette[:len(ndf.columns)]
    else :
        colors = None

    if "color" in kwargs : kwargs.pop("color")
    kind = "barh" if horizontal else "bar"

    ndf.plot(* args, kind=kind, stacked=True, ax=ax, color=colors, ** kwargs)

def stacked_barplot_diff(x, y, hue, data, ax, aggfunc=None, palette=None, ** kwargs) :

    df, dfs = data, {}
    aggfunc = aggfunc or np.mean
    palette = palette or sns.color_palette()

    if isinstance(palette, dict) : palette = {nhue : palette[nhue] for nhue in sorted(df[hue].unique())}
    else : palette = {nhue : palette[idx] for idx, nhue in enumerate(sorted(df[hue].unique()))}

    for name, sdf in df.groupby(x) :
        serie = sdf.groupby(hue)[y].apply(aggfunc)
        serie = serie.sort_values()
        index = tuple(serie.index)
        serie = serie.diff().fillna(serie.min())
        
        sdf = serie.rename(y).reset_index()
        sdf[x] = name
        dfs.setdefault(index, []).append(sdf)

    zero_df = pd.concat(itertools.chain(* dfs.values()))
    zero_df[y] = 0
    start = True

    x = [x] if not isinstance(x, list) else x
    hue = [hue] if not isinstance(hue, list) else hue

    for hueorder, sdfs in dfs.items() :
        df = pd.concat(sdfs)
        df = pd.concat((df, zero_df)).drop_duplicates(x + hue).sort_values(x + hue)

        legend = start
        start = False
        
        stacked_barplot(x=x, y=y, hue=hue, data=df, stack_order=list(hueorder), 
            palette=palette, ax=ax, legend=legend, ** kwargs)

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

def barplot(data, ax, hue, * args, ** kwargs) :
    colors = kwargs.pop("palette", None)
    sns.barplot(* args, data=data, ax=ax, ** kwargs)
    shues = sorted(data[hue].unique())
    hues = list(data[hue])
    
    if colors is None : 
        colors = sns.color_palette()
        colors = {hue : colors[idx] for idx, hue in enumerate(shues)}

    for idx, patch in enumerate(ax.patches) :
        hue = hues[idx]
        try : color = colors[hue]
        except KeyError : raise KeyError("Hue color not found in the palette : '%s'" %(str(hue)))
        patch.set_color(color)

    graph_utils.add_custom_basic_legend(ax, shues, colors)

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

def non_linear_reg(fun, col1, col2, data, ax, kwargs_cfit={}, yvalues_plot=None, nid=None, * args, ** kwargs) :
    # https://stackoverflow.com/questions/46497892/non-linear-regression-in-seaborn-python
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # Non linear regression using scipy
    # You have to provide a function which will take X, A, B and C as arguments

    c1, c2 = np.array(data[col1]), np.array(data[col2])
    popt, pcov = curve_fit(fun, c1, c2, ** kwargs_cfit)
    
    # r2 calculation
    # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    
    residuals = c2 - fun(c1, * popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((c2 - np.mean(c2)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    c1 = yvalues_plot if yvalues_plot is not None else c1
    yvalues = fun(c1, * popt)
    
    if not yvalues.any() : raise Exception("Unable to find yvalues with this law")
    ax.plot(c1, yvalues, * args, ** kwargs)

    res = {estimator : popt[idx] for idx, estimator in enumerate("abc")}
    res["r2"] = r2
    res["nlr_y"] = yvalues

    return res

def draw_diagonal(data, ax, ** kwargs):
    kwg = {"linestyle" : "--", "linewidth" : .5, "color" : "#ca472f", 'alpha': .5}
    kwg.update(kwargs)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ** kwg)

def corrplot(data, ax, x, y, fun=None, law="pearson", diagonal=True, annotate=True, share_lim=True,
        diag_kwg={}, text_fun=None, text_kwg={}, ** kwargs) :

    # Plot correlation data and annotate using pearson or spearman correlation value
    funs = {"pearson" : pearsonr, "spearman" : spearmanr}
    if annotate and law not in funs :
        raise Exception("Law must be either pearson of spearman")

    fun = fun or sns.regplot
    fun(x=x, y=y, ax=ax, data=data, ** kwargs)

    if share_lim :
        min_lim = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1])

        ax.set_xlim((min_lim, max_lim))
        ax.set_ylim((min_lim, max_lim))

    if diagonal:
        draw_diagonal(data, ax, ** diag_kwg)

    default_text_fun = lambda law, coef, pvalue: f'{law} coefficient : {coef:.2f}\nP-value : {pvalue:.2E}'

    if annotate :
        x = data[x]
        y = data[y]

        fun = funs[law]
        coef, pvalue = fun(x, y)

        fun = text_fun or default_text_fun
        line = fun(law, coef, pvalue)

        kwg = {"x" : 0.05, "y" : 0.9, "s" : line, "ha" : "left", "va": "top", "transform" : ax.transAxes}
        kwg.update(text_kwg)
        ax.text(** kwg)

def remove_non_number(df, columns) :
    # used by scatterplot
    for column in columns :
        df = df[np.isfinite(df[column])]
    return df

def scatterplot(data, ax, col1, col2, ccol=None, ccolname=None, hue=None, huecol=None, titlehue=True, 
        hue_order=None, kws_scatter={}, kwg_corr={}, ** kwg_regplot) :
        
    # To use until I change my seaborn version
    # since they add a scatterplot function now : https://seaborn.pydata.org/generated/seaborn.scatterplot.html

    df = data
    df = remove_non_number(df, (col1, col2))

    if ccol is not None :
        # http://stackoverflow.com/questions/13943217/how-to-add-colorbars-to-scatterplots-created-like-this
        cmap = sns.cubehelix_palette(light=.9, as_cmap=True)
        df = remove_non_number(df, (ccol, ))
        third_variable = df[ccol]
        skws = {"c" : third_variable, "cmap" : cmap, "color" : None}
        skws.update(kws_scatter)
        sns.regplot(col1, col2, data=df, ax=ax, scatter_kws=skws, ** kwg_regplot)

        outpathc = [child for child in ax.get_children()
        if isinstance(child, PathCollection)][0]
        cbar = plt.colorbar(mappable=outpathc, ax=ax)
        ccolname = ccol if ccolname is None else ccolname
        cbar.set_label(ccolname)

    elif hue is not None:

        color = kwg_regplot.pop("palette", None)
        if color is not None and not huecol :
            huecol = {hname : color[idx] for idx, hname in enumerate(df[hue].unique())}

        hue_order = hue_order or sorted(df[hue].unique())
        for hue_value in hue_order :
            sdf = df[df[hue] == hue_value]
            color = huecol[hue_value] if huecol and hue_value in huecol else None
            sns.regplot(col1, col2, data=sdf, ax=ax, label=hue_value, color=color, ** kwg_regplot)
        
        ax.legend()

    else :
        sns.regplot(col1, col2, data=df, ax=ax, ** kwg_regplot)

    return df[[col1, col2]].corr(** kwg_corr).iat[0,1]

def gplot(data, ax, start="start", end="end", strand=None, kind=None, name=None, legend=True, palette={}, kwargs_arrow={}, kwargs_text={}) :
    
    df = data

    kwargs_arrow = {
        "length_includes_head" : True,
        "width" : .5,
        "head_width" : .5,
        ** kwargs_arrow
    }

    pmin = min((df[start].min(), df[end].min()))
    pmax = max((df[start].max(), df[end].max()))

    if kind : palette = palette or gplot_default_cpal(df, kind)
    tracks = []

    for idx, row in data.iterrows() :
        fstart, fend = row[start], row[end]
        if fstart > fend : fstart, fend = fend, fstart
        fstrand = row.get(strand, strand)
        track_idx = find_track(tracks, fstart, fend)

        fkind = row.get(kind, kind)
        color = palette.get(fkind, None)
        fname = row.get(name, name)

        plot_feature(ax, fstart, fend, fstrand, fname, color, track_idx, kwargs_arrow, kwargs_text)
  
    ax.set_xlim((pmin, pmax))
    ax.set_ylim((-.5, len(tracks)))
    ax.set_yticks([])

    if palette and legend :
        graph_utils.basic_legend(ax, palette, loc=2)

def gplot_default_cpal(df, kind) :
    cpal = sns.color_palette()
    return {kind : cpal[idx] for idx, kind in enumerate(sorted(df[kind].unique()))}

def find_track(tracks, start, end) :

    idx = - 1
    for idx, track in enumerate(tracks) :
        found = False 

        for feature in track :
            fstart, fend = feature
            
            if fstart <= start <= fend or fstart <= end <= fend :
                found = True
                break

        if not found :
            track.append((start, end))
            return idx

    tracks.append([(start, end)])
    return idx + 1

def plot_feature(ax, start, end, strand, name, color, track_idx, kwargs_arrow, kwargs_text) :
    
    size = end - start
    headlen = size * .2 
    center = (start + size / 2)

    x = start if strand == "+" else end
    dx = size if strand == "+" else - size

    color = color or constants.DEFAULT_COLOR

    ax.arrow(x, track_idx, dx, 0, head_length=headlen, color=color, ** kwargs_arrow)

    if name :
        ax.text(center, track_idx + .5, name, color="black", 
            horizontalalignment="center", verticalalignment="center", ** kwargs_text)


class PairwiseHeatmap() :

    # It is NOT a graph object but simply
    # a simple object to make the code cleaner
    # instead of putting everything inside a function
    # use pairwise_heatmap function if you want to use it

    def __init__(self, data, key, value, ax, intersection=True, ** kwargs) :

        self.df = data
        self.key = key
        self.value = value
        self.ax = ax

        self.sim = self.get_sim(intersection)
        self.plot(self.sim, ** kwargs)

    def get_sim(self, intersection) :

        groups = sorted(self.df[self.value].unique())
        values = np.zeros((len(groups), len(groups)), dtype=int)
        get_keys = lambda x : set(self.df[self.df[self.value] == x][self.key])

        for g1, g2 in itertools.combinations(groups, 2) :
            if intersection : ncount = len(get_keys(g1) & get_keys(g2))
            else : ncount = len(get_keys(g1) | get_keys(g2))
            
            idx1, idx2 = groups.index(g1), groups.index(g2)
            if idx1 > idx2 : idx1, idx2 = idx2, idx1
            values[idx1][idx2] = ncount

        for value, sdf in self.df.groupby(self.value) :
            idx = groups.index(value)
            values[idx][idx] = sdf[self.key].nunique()

        return pd.DataFrame(values, index=groups, columns=groups)

    def plot(self, df, ** kwargs) :
        sns.heatmap(data=df, ax=self.ax, ** kwargs)

def pairwise_heatmap(value, hue, data, ax, ** kwargs) :
    return PairwiseHeatmap(data, value, hue, ax, ** kwargs)


def pie(x, data, ax, labels=None, explode={}, equal=True, colors=None, ** kwargs) :
        # A wrapper for matplotlib ax.pie but using a dataframe as input

        kwargs.setdefault("x", data[x])

        # colors can be a dict, usefull if scontainer is used
        if colors :
            colors = graph_utils.colors_from_arg(colors, data, labels)
            kwargs["colors"] = colors

        # in case of scontainer usage
        # color argument is provided by scontainer process
        kwargs.pop("color", None)

        if labels :
            kwargs.setdefault("labels", data[labels])
            kwargs.setdefault("explode", [explode.get(label, 0) for label in data[labels]])
        
        kwargs.setdefault("autopct", "%1.1f%%")
        kwargs.setdefault("startangle", 90)

        ax.pie(** kwargs)
        if equal : ax.axis('equal')


def donut(var, value, ax, data, width=.2, opening=0.1, palette=None, ** kwargs) :
    labels, values = list(data[var]), list(data[value])
    if palette : colors = palette[:len(labels)]
    else : colors = sns.color_palette()[:len(labels)] 
    
    if opening :
        if opening >= 1 : raise Exception("Opening value must be < 1")
        opening_value = int(sum(values) * opening)
        labels.append("")
        values.append(opening_value)
        colors.append("white")
    
    ax.pie(values, labels=labels, colors=colors, ** kwargs)

    circle = Circle( (0,0), 1-width, color='white')
    ax.add_artist(circle)
