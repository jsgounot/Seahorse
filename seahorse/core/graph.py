# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-03-29 15:55:41
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 17:18:38

from matplotlib.ticker import FuncFormatter

from seahorse import sns
from seahorse.core.figure import Fig
from seahorse.core.gwrap import LibWrapper
from seahorse.core import graphfun

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

    def apply_xticklabels(self, fun, which="major", ** kwargs) :
        ufun = (lambda x : fun.get(x, x)) if isinstance(fun, dict) else fun
        self.ax.set_xticklabels(list(self._ticks_names(self.ax.get_xticklabels(which=which), ufun)), ** kwargs)

    def transform_xticklabels(self, fun, which="major", ** kwargs) :
        self.ax.set_xticklabels(list(fun(self.ax.get_xticklabels(which=which))), ** kwargs)
                
    def transform_xticks(self, fun, ** kwargs) :
        self.ax.set_xticks(list(fun(self.ax.get_xticks())), ** kwargs)

    def apply_ytickslabels(self, fun, which="major", ** kwargs) :
        ufun = (lambda x : fun.get(x, x)) if isinstance(fun, dict) else fun
        self.ax.set_yticklabels(list(self._ticks_names(self.ax.get_yticklabels(which=which), ufun)), ** kwargs) 

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
        kwargs.setdefault("prop", {"size" : 15})
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