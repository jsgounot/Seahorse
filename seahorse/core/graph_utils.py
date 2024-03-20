# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-05-16 13:53:18
# @Last modified by:   jsgounot
# @Last Modified time: 2024-03-20 14:12:04

try: 
    from collections import Mapping, Iterable
except ImportError:
    from collections.abc import Mapping, Iterable
  
from numpy import isfinite, isnan, pi

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Ellipse

import seaborn as sns

def remove_non_number(df, columns) :
    df = df.dropna(axis=0, how="any")

    # for now I don't know how to solve this
    # but must be fixed soon (hopefully)

    """    
    for column in columns :
        print (df[column])
        df = df[isfinite(df[column])]
    """

    return df

# -------------------------
# Graph dist

class GraphDist() :
    # Graphical distance in a plot
    # Used to plot dot in graph (see set_colored_dot)
    # http://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals/41468977#41468977

    def __init__(self, size, ax, x=True) :
        self.size = size
        self.ax = ax
        self.x = x

    @property
    def dist_real(self) :
        x0, y0 = self.ax.transAxes.transform((0, 0)) # lower left in pixels
        x1, y1 = self.ax.transAxes.transform((1, 1)) # upper right in pixes
        value = x1 - x0 if self.x else y1 - y0
        return value

    @property
    def dist_abs(self) :
        bounds = self.ax.get_xlim() if self.x else self.ax.get_ylim()
        return bounds[1] - bounds[0]

    @property
    def value(self) :
        return (self.size / self.dist_real) * self.dist_abs

    @property
    def pvalue(self) :
        return (self.dist_abs / self.dist_real) * self.size

    @property
    def min_ax_value(self) :
        bounds = self.ax.get_xlim() if self.x else self.ax.get_ylim()
        return min(bounds)

    def __mul__(self, obj) :
        # used for ellipse width
        return self.value * obj

    def __float__(self) :
        # used for ellipse position
        return self.min_ax_value + self.pvalue

def label_dots(ax, dic={}, xaxis=True, size=12, axgap=10, padticks=20, defaultcolor="#ffffff") :
    # Combine two methods
    # In this function we do not change the label we draw a dot directly in the graph
    # If we want the dot to be directly in the label, an other solution using latex could work
    # This one use a kind of custom int object GraphDistance calculating the correct ellipse width
    # and height whenever the ax and the graph dimensions change
    # http://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
    # http://stackoverflow.com/questions/11995148/plot-circle-on-unequal-axes-with-pyplot

    ticks = ax.get_xticklabels() if xaxis else ax.get_yticklabels()
    ticks_locations = ax.get_xticks() if xaxis else ax.get_yticks()

    width = GraphDist(size, ax, True)
    height = GraphDist(size, ax, False)
    fdic = {}

    for index, tick in enumerate(ticks) :
        xposition = ticks_locations[index] if xaxis else GraphDist(0 - axgap, ax, True)
        yposition = GraphDist(0 - axgap, ax, False) if xaxis else ticks_locations[index]
        color = dic.get(tick.get_text(), defaultcolor)
        circ = Ellipse((xposition, yposition), width, height, clip_on=False, facecolor=color)
        ax.add_artist(circ)
        fdic[tick.get_text()] = circ

    ax.tick_params(pad=padticks)
    return fdic

# -------------------------
# Colors

def cmap_from_color(colors) :
    return ListedColormap(colors)

def color_palette(* args, ** kwargs) :
    return sns.color_palette(* args, ** kwargs)

def color_palette_mapping(elements, * args, ** kwargs) :
    cpal = color_palette(* args, ** kwargs)
    elements = sorted(set(elements))
    return {element : cpal[idx] for idx, element in enumerate(elements)}

def palette_ten() :
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", 
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    return sns.color_palette(colors)

def colors_from_arg(colors, df, column) :
    # mimic all possibilites from colors argument from seahorse
    # something which should be used in all function ?

    error_message = """ Colors can be either a string (dataframe column name),
    a mapping object (correspondance to x values) or an iterable (colors list)
    """

    if isinstance(colors, str) :
        if colors == column :
            colors = color_palette_mapping(df[column])
        elif colors in df.columns :
            return list(df[colors])
        else :
            raise ValueError(error_message)

    elif isinstance(colors, Mapping) :
        return [colors[element] for element in df[column]]

    elif isinstance(colors, Iterable) :
        return colors

    else :
        raise ValueError(error_message)


# -------------------------
# Legend


def basic_legend(ax, names_color, * args, ** kwargs) :
    handle = [Patch(facecolor=color, edgecolor=color, label=name)
        for name, color in names_color.items()]
    
    handle = sorted(handle, key = lambda patch : patch.get_label())

    ax.legend(handles=handle, * args, ** kwargs)

def add_custom_basic_legend(ax, names, palette=None, ** kwargs) :
    palette = sns.color_palette() if palette is None else palette
    if isinstance(palette, dict) : palette = [palette[name] for name in names]
    patches = [Patch(color=palette[idx], label=label)
    for idx, label in enumerate(names)]
    ax.legend(handles = patches, ** kwargs)
