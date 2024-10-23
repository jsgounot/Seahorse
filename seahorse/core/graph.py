# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-03-29 15:55:41
# @Last modified by:   jsgounot
# @Last Modified time: 2024-10-23 10:34:58

import numpy as np
from itertools import combinations

from matplotlib.ticker import FuncFormatter
from matplotlib.patches import PathPatch

from seahorse import sns
from seahorse.core.figure import Fig
from seahorse.core.gwrap import LibWrapper
from seahorse.core import graphfun

try:
    from statannotations.Annotator import Annotator
except ImportError:
    print ('Unable to import Annotator from statannotations (https://github.com/trevismd/statannotations), some features might not work')

class Graph(Fig) :

    def __init__(self, data=None, ax=None, copy=False, ** kwargs) :
        data = copy(data) if copy else data

        if ax is None :
            self.create_fig()
            self.ax = self.fig.add_subplot(111, ** kwargs)
        
        else :
            self.ax =  ax
            self.fig = ax.get_figure()

        self.update_data(data)

    def update_data(self, data) :
        self.data = data
        self.df =  LibWrapper(self.data, ax = self.ax)
        self.sns = LibWrapper(sns, data = self.data, ax = self.ax)
        self.shs = LibWrapper(graphfun, data = self.data, ax = self.ax)

    """
    Global apply
    """

    def apply(self, fun, * args, ** kwargs) :
        fun(self, * args, ** kwargs)

    """
    Labels, title and ticks
    """

    def set_labels(self, xlabel=None, ylabel=None, ** kwargs) :
        kwargs.setdefault("size", 15)
        if xlabel is not None : self.ax.set_xlabel(str(xlabel), ** kwargs)
        if ylabel is not None : self.ax.set_ylabel(str(ylabel), ** kwargs)

    def _ticks_names(self, ticks_iterator, fun) :
        for tick in ticks_iterator :
            text = tick.get_text()
            yield fun(text)

    def apply_xticklabels(self, fun={}, which="major", ** kwargs) :
        ufun = (lambda x : fun.get(x, x)) if isinstance(fun, dict) else fun
        labels = list(self._ticks_names(self.ax.get_xticklabels(which=which), ufun))
        ticks = self.ax.get_xticks()
        self.ax.set_xticks(ticks, labels=labels, ** kwargs)

    def transform_xticklabels(self, fun, which="major", ** kwargs) :
        self.ax.set_xticklabels(list(fun(self.ax.get_xticklabels(which=which))), ** kwargs)
                
    def transform_xticks(self, fun, ** kwargs) :
        self.ax.set_xticks(list(fun(self.ax.get_xticks())), ** kwargs)

    def apply_yticklabels(self, fun={}, which="major", ** kwargs) :
        ufun = (lambda x : fun.get(x, x)) if isinstance(fun, dict) else fun
        labels = list(self._ticks_names(self.ax.get_yticklabels(which=which), ufun))
        ticks = self.ax.get_yticks()
        self.ax.set_yticks(ticks, labels=labels, ** kwargs)

    def transform_yticklabels(self, fun={}, which="major", ** kwargs) :
        self.ax.set_yticklabels(list(fun(self.ax.get_yticklabels())), ** kwargs)

    def transform_yticks(self, fun, ** kwargs) :
        self.ax.set_yticks(list(fun(self.ax.get_yticks())), ** kwargs)

    def set_ticks_windows(self, window, * args, ** kwargs) :

        # should use either ticks position or ticks values and not only value here !
        # also rename and move this function 

        labels = []

        for label in self.ax.get_xticklabels() :
            text = label.get_text()
            try :
                if not float(text) % window :
                    labels.append(text)
                else :
                    labels.append("")
            except ValueError :
                labels.append(text)

        self.ax.set_xticklabels(labels, * args, ** kwargs)

    def remove_xticks(self, ** kwargs) :
        kwargs = {"axis" : "x", "which" : "both", "bottom" : False,
            "top" : False, "labelbottom" : False, "labeltop" : False, 
            ** kwargs}
        
        self.ax.tick_params(** kwargs)

    def remove_yticks(self, ** kwargs) :
        kwargs = {"axis" : "y", "which" : "both", "right" : False,
            "left" : False, "labelleft" : False, "labelright" : False, 
            ** kwargs}
        
        self.ax.tick_params(** kwargs)

    """
    Legends
    """

    def remove_legend(self) :
        if self.ax.legend_ is None : return
        self.ax.legend_.remove()

    def set_legend(self, * args, ** kwargs) :
        self.ax.legend(* args, ** kwargs)

    def legend_size(self, size) :
        self.set_legend(prop={"size" : size})

    def get_legend(self) :
        return self.ax.legend_

    def apply_legend(self, fun, * args, ** kwargs) :
        l = self.get_legend()
        if l is None : return
        getattr(l, fun)(* args, ** kwargs)

    def legend_outside(self, ** kwargs) :
        kwargs.setdefault("loc", "upper left")
        kwargs.setdefault("bbox_to_anchor", (1, 1))
        #kwargs.setdefault("prop", {"size" : 15})
        self.set_legend(** kwargs)

    """
    Scale
    """

    def _format_axis_ticks(self, func, axis="x") :
        if axis == "x" : self.ax.get_xaxis().set_major_formatter(FuncFormatter(func))
        elif axis =="y" : self.ax.get_yaxis().set_major_formatter(FuncFormatter(func))
        else : raise ValueError("axis should be 'x' or 'y'")

    def thousand_sep(self, axis="x", separator=",") :
        func = lambda x, p : format(int(x), separator)
        self._format_axis_ticks(func, axis)

    """
    Aesthetic
    """

    def set_ax_lim(self, x=None, y=None) :
        if x : self.ax.set_xlim(x)
        if y : self.ax.set_ylim(y)

    def share_ax_lim(self) :
        min_lim = min(self.ax.get_xlim()[0], self.ax.get_ylim()[0])
        max_lim = max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])

        self.ax.set_xlim((min_lim, max_lim))
        self.ax.set_ylim((min_lim, max_lim))

    def adjust_edge_margins(self, edge, prc=True, value=0.01) :
        if edge not in ["top", "bot", "left", "right"] :
            raise ValueError("Edge must be top, bot, left or right")

        start, end = self.ax.get_ylim() if edge in ["top", "bot"] else self.ax.get_xlim()
        margin_value = (end - start) * value
        idx = 0 if edge in ["bot", "left"] else 1
        values = [start, end]
        values[idx] = values[idx] + margin_value if prc else values[idx] + value
        setter = self.ax.set_ylim if edge in ["top", "bot"] else self.ax.set_xlim
        setter(tuple(values))

    def adjust_hori_margins(self, prc_margin=0.01) :
        start, end = self.ax.get_xlim()
        horisize = end - start
        margin_value = horisize * prc_margin
        self.ax.set_xlim((start - margin_value, end + margin_value))

    def adjust_vert_margins(self, prc_margin=0.01) :
        start, end = self.ax.get_ylim()
        vertsize = end - start
        margin_value = vertsize * prc_margin
        self.ax.set_ylim((start - margin_value, end + margin_value))

    def adjust_margins(self, prc_margin=0.01) :
        self.adjust_vert_margins(prc_margin)
        self.adjust_hori_margins(prc_margin)

    """
    barplot
    """

    def barplot_add_value(self, asint=False, rotation=0, asprc=False, both=False, kwg_text={}, fntxt=lambda x : x, idxs=None) :
        spacer = self.ax.get_ylim()[1] * 2/100.
        sum_v = float(sum([p.get_height() for p in self.ax.patches]))
        patches = sorted(self.ax.patches, key = lambda patch : patch.get_x())

        for idx, p in enumerate(patches) :
            if idxs and idx not in idxs : continue

            height = p.get_height()
            height_name = str(int(height)) if asint and not np.isnan(height) else height
            height_name = "%.1f" %(height * 100 / sum_v) if asprc else height_name
            height_name = height_name + "\n(%.1f" %(height * 100 / sum_v) + "%)" if both else height_name
            height_name = fntxt(height_name)
            x = p.get_x() + p.get_width() / 2.
            self.ax.text(x, height + spacer, str(height_name), ha="center",
            va="bottom", rotation=rotation, **kwg_text)


    """
    Annotator
    """

    @staticmethod
    def get_pairs(data, main, hue=None):
        mains = data[main].sort_values().unique()
        
        if hue is not None:
            values = data.groupby(main)[hue].unique().apply(list).to_dict()
            return (((main, hue_x), (main, hue_y))
                    for main in mains
                    for hue_x, hue_y in combinations(values[main], 2))
        
        else:
            return combinations(mains, 2)
        
    def make_annot(self, x, y, hue=None, data=None, pairs=None, orient='v', test='Mann-Whitney', ** kwargs):
        data = data if data is not None else self.data

        if data is None:
            raise Exception("Graph data can't be set to None")

        main = x if orient =='v' else y
        pairs = Graph.get_pairs(data, main, hue) if pairs is None else pairs
        
        order = kwargs.pop('order', None)
        pairs = list(pairs)    
            
        annotator = Annotator(self.ax, pairs, data=data, x=x, y=y, hue=hue, order=order, orient=orient)
        annotator.configure(test=test, ** kwargs)
        annotator.apply_and_annotate()

    """
    utility functions
    """

    def add_xticks_ncount(self, column, df=None, fun=None) :
        df = df if df is not None else self.data
        counts = df.groupby(column).size().to_dict()
        counts = {str(key): value for key, value in counts.items()}
        nticks = []
        for element in self.ax.get_xticklabels() :
            name = element.get_text()
            count = counts[name]
            if fun :
                new = fun(name, count)
            else :
                new = name + "\n(N=%i)" %(count)
            nticks.append(new)
        self.ax.set_xticklabels(nticks)

    def add_yticks_ncount(self, column, df=None, fun=None) :
        df = df if df is not None else self.data
        counts = df.groupby(column).size().to_dict()
        counts = {str(key): value for key, value in counts.items()}
        nticks = []
        for element in self.ax.get_yticklabels() :
            name = element.get_text()
            count = counts[name]
            if fun :
                new = fun(name, count)
            else :
                new = name + "\n(N=%i)" %(count)
            nticks.append(new)
        self.ax.set_yticklabels(nticks)

    def change_bars_width(self, new_prop) :
        for patch in self.ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_prop

            # we change the bar width
            patch.set_width(new_prop)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    def change_boxplot_width(self, fac=.9) :
        # https://github.com/mwaskom/seaborn/issues/1076
        # https://stackoverflow.com/questions/56838187/how-to-create-spacing-between-same-subgroup-in-seaborn-boxplot

        # TODO : Only work for horrizontal boxplot
        # Need to be adjusted in case of vertical ones

        for c in self.ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin+xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in self.ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])