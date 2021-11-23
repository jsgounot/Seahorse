# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-03-29 15:52:33
# @Last modified by:   jsgounot
# @Last Modified time: 2021-11-23 10:15:43

from itertools import product
import matplotlib.gridspec as gridspec

import pandas as pd

from seahorse.core import graphfun
from seahorse import sns

from seahorse.core.figure import Fig
from seahorse.core.gwrap import GraphAttributes, GBLibWrapper, GroupByPlotter
from seahorse.core.graph import Graph

class MinimalSlice() :

    def __init__(self, values) :
        if not isinstance(values, slice) :
            raise ValueError("Parameter should be a slice")
         
        self.start, self.stop = values.start, values.stop

        # We remove None values
        self.start = self.stop if self.start is None else self.start
        self.stop = self.start if self.stop is None else self.stop

        if self.start is None :
            raise ValueError("At least the start or the stop must be provided in the slice")

    def __contains__(self, value) :
        return value in range(self.start, self.stop)

    def __eq__(self, value) :
        if isinstance(value, MinimalSlice) :
            return self.start == value.start and self.stop == value.stop
        return False

    def __str__(self) :
        return str((self.start, self.stop))

    def __repr__(self) :
        return str(self)

    def __hash__(self) :
        return hash((self.start, self.stop))

    def as_slice(self):
        return slice(self.start, self.stop)
    
    def isin(self, other) :
        if not isinstance(other, MinimalSlice) :
            raise ValueError("Overlapp can only be calculated comparing two MinimalSlice")

        check_one = other.start <= self.start < other.stop
        check_two = other.start < self.stop < other.stop
        return check_one or check_two

    def overlapp(self, other) :
        return self.isin(other) or other.isin(self)

class SliceTuple() :

    def __init__(self, row, col) :
        self.ms_row = MinimalSlice(row)
        self.ms_col = MinimalSlice(col)

    def __str__(self) :
        return str((self.ms_row, self.ms_col))

    def __repr__(self) :
        return str(self)

    def __hash__(self) :
        return hash((hash(self.ms_row), hash(self.ms_col)))

    def __eq__(self, value) :
        if not isinstance(value, SliceTuple) :
            return False

        return self.ms_row == value.ms_row and self.ms_col == value.ms_col

    def as_slices(self) :
        return (self.ms_row.as_slice(), self.ms_col.as_slice())

    def overlapp(self, other) :
        if not isinstance(other, SliceTuple) :
            raise ValueError("Overlapp can only be calculated comparing two SliceTuple")

        return self.ms_row.overlapp(other.ms_row) and self.ms_col.overlapp(other.ms_col)

class GSManager() :

    def __init__(self, gridspec_instance, ax_call) :
        # ss : SubplotSpec
        self._gs = gridspec_instance
        self._axes = {}
        self.ax_call = ax_call

    @property
    def shape(self):
        return self._gs.get_geometry()

    @property
    def nrows(self) :
        return self.shape[0]

    @property
    def ncols(self) :
        return self.shape[1]
    
    def __iter__(self) :
        return iter(self._axes.values())

    def sorted(self) :
        sfun = lambda x : (x.ms_row.start, x.ms_col.start)
        keys =  sorted(self._axes, key=sfun)
        return (self._axes[key] for key in keys)
    
    def transform_index(self, index) :
        # Return a slice tuple from an index
        # Should be either an integer or a tuple of size 2
        # containing either integers or slices

        if isinstance(index, int) :
            nvalue, cvalue = index, index
            nvalue = slice(nvalue // self.ncols, (nvalue // self.ncols) + 1)            
            cvalue = slice(cvalue % self.ncols, (cvalue % self.ncols) + 1)

        else :
            nvalue, cvalue = index
            if isinstance(nvalue, int) : nvalue = slice(nvalue, nvalue + 1)
            if isinstance(cvalue, int) : cvalue = slice(cvalue, cvalue + 1)
        
        return SliceTuple(nvalue, cvalue)

    def get(self, index, default=None) :
        st = self.transform_index(index)
        return self._axes.get(st, default)

    def get_make(self, index, check=True) :
        st = self.transform_index(index)

        if st in self._axes :
            return self._axes[st]

        # Check if the slice overlapp an other slice
        if check and any(st.overlapp(ost) for ost in self._axes) :
            raise Exception("The current index overlapp a previous index")

        ss = self._gs[st.as_slices()]
        ax = self.ax_call(ss)
        return self._axes.setdefault(st, ax)

class SubplotsContainer(Fig) :
    
    def __init__(self, data, * args, ** kwargs) :
        self.data = data
        self.gsm = {}
        self.used_gs = "base"

        self.create_fig()
        self.add_gs("base", * args, ** kwargs)
        self.setup_ax_labels()

    @property
    def cgs(self):
        return self.gsm[self.used_gs]

    @property
    def cgs_shape(self):
        return self.cgs.shape
    
    @property
    def axes(self):
        return iter(self.cgs)
    
    def gs_update(self, * args, gname="base", ** kwargs) :
        self.gs.update(* args, ** kwargs)

    def ax(self, idx, gname=None) :
        gname = gname or self.used_gs
        return self.gsm[gname].get_make(idx)

    def graph(self, idx, data=None) :
        data = self.data if data is None else data
        return Graph(data, self.ax(idx))

    def setup_ax_labels(self) :
        self.ax_labels = self.fig.add_subplot(111, frameon=False)
        self.ax_labels.set_xticks([])
        self.ax_labels.xaxis.labelpad = 30
        self.ax_labels.set_yticks([])
        self.ax_labels.yaxis.labelpad = 30
        self.ax_labels.grid(False)

    def __getitem__(self, items) :
        self.used_gs = items
        return self   

    def add_subplot(self, ss, ** kwargs) :
        return self.fig.add_subplot(ss, ** kwargs)

    def add_gs(self, name, * args, ** kwargs) :
        gs = gridspec.GridSpec(* args, ** kwargs)
        self.gsm[name] = GSManager(gs, self.add_subplot)

    # ---------------------------------------------------------------
    # Utility functions

    def fill_combinations(df, columns, cvalue, value) :
        if len(columns) < 2 : raise ValueError("columns length must higher than 1")
        data = []
        for combination in product(*[df[column] for column in columns]) :
            data.append({cvalue : value, ** {column : combination[idx] for idx, column in enumerate(columns)}})
        df = pd.concat((df, pd.DataFrame(data)))
        return df.drop_duplicates(columns)
  
    # ---------------------------------------------------------------

    def fill_axes(self) :
        nrow, ncol = self.cgs_shape
        return [self.ax(i) for i in range(nrow * ncol)]

    def get_edges(self, top=False, bottom=False, left=False, right=False, 
            asgraph=False, reverse=False, union=True) :
        
        if reverse : 
            axes = set(self.get_edges(top=top, bottom=bottom, left=left, right=right))

            for ax in self.cgs :
                if ax in axes : continue
                yield Graph(ax=ax) if asgraph else ax

            raise StopIteration

        for ax in self.cgs :
            res = []

            if top : res.append(ax.get_subplotspec().is_first_row())
            if bottom : res.append(ax.get_subplotspec().is_last_row())
            if left : res.append(ax.get_subplotspec().is_first_col())
            if right : res.append(ax.get_subplotspec().is_last_col())

            if union == True and all(res) != True : continue
            if union == False and any(res) != True : continue

            yield Graph(ax=ax) if asgraph else ax

    def groupby(self, hue, ** kwargs) :
        return GroupByPlotter(self.data, self, hue, ** kwargs)

    def apply(self, graph=True) :
        return GraphAttributes(list(self.cgs), graph)

    def clean_graph_labels(self) :
        ax_left = list(self.get_edges(left=True))
        ax_bottom = list(self.get_edges(bottom=True))

        for ax in self.cgs :
            if ax in ax_left and ax in ax_bottom : continue
            elif ax in ax_left : ax.set_xlabel("")
            elif ax in ax_bottom : ax.set_ylabel("")
            else : ax.set_ylabel(""); ax.set_xlabel("")

    def sharex(self, ignored_axes=[], rm_ticks=True) :
        ax_bottom = list(self.get_edges(bottom=True))
        axes = [ax for ax in self.cgs if ax not in ignored_axes]

        min_xlim = min(min(ax.get_xlim()) for ax in axes)
        max_xlim = max(max(ax.get_xlim()) for ax in axes)
        
        for ax in axes :
            ax.set_xlim((min_xlim, max_xlim))
            if ax not in ax_bottom and rm_ticks : 
                ax.set_xticklabels(["" for _ in ax.get_xticklabels()])
                ax.set_xlabel("")

    def sharey(self, ignored_axes=[], rm_ticks=True) :
        ax_left = list(self.get_edges(left=True))
        axes = [ax for ax in self.cgs if ax not in ignored_axes]

        min_ylim = min(min(ax.get_ylim()) for ax in axes)
        max_ylim = max(max(ax.get_ylim()) for ax in axes)

        for ax in axes :
            ax.set_ylim((min_ylim, max_ylim))
            if ax not in ax_left and rm_ticks : 
                ax.set_yticklabels(["" for _ in ax.get_yticklabels()])
                ax.set_ylabel("")

    def share_axes(self, ignored_axes=[], rm_ticks=True) :
        self.sharex(ignored_axes, rm_ticks)
        self.sharey(ignored_axes, rm_ticks)

    def set_labels(self, xlabel=None, ylabel=None, sub=True, ** kwargs) :
        if sub : self._set_labels_sub(xlabel, ylabel, ** kwargs)
        else : self._set_labels_main(xlabel, ylabel, ** kwargs)

    def _set_labels_main(self, xlabel=None, ylabel=None, ** kwargs) :
        if xlabel : self.ax_labels.set_xlabel(xlabel, ** kwargs)
        if ylabel : self.ax_labels.set_ylabel(ylabel, ** kwargs)

    def _set_labels_sub(self, xlabel=None, ylabel=None, ** kwargs) :
        ax_left = list(self.get_edges(left=True))
        ax_bottom = list(self.get_edges(bottom=True))

        for ax in self.cgs :
            ax_xlabel = xlabel if ax in ax_bottom else ""
            ax_ylabel = ylabel if ax in ax_left else ""

            ax.set_xlabel(ax_xlabel, ** kwargs)
            ax.set_ylabel(ax_ylabel, ** kwargs)

    def select_legend(self, index, topright=False, ** kwargs) :

        if topright :
            kwargs.setdefault("loc", "upper left")
            kwargs.setdefault("bbox_to_anchor", (1, 1))
            kwargs.setdefault("prop", {"size" : 15})

        for idx, ax in enumerate(self.cgs.sorted()) :
            if idx == index and kwargs :
                self.graph(idx).set_legend(** kwargs)
            elif idx != index :
                self.graph(idx).remove_legend()

class FacetGrid(SubplotsContainer) :

    def __init__(self, data, x, y) :
        self.xname, self.yname = x, y
        self.xnames = tuple(sorted(data[x].unique()))
        self.ynames = tuple(sorted(data[y].unique()))
        super().__init__(data, len(self.xnames), len(self.ynames))

        self.shs = GBLibWrapper(graphfun, self, self.gbp_iterator, True, True, False, ax_by_name=True)
        self.sns = GBLibWrapper(sns, self, self.gbp_iterator, True, True, False, ax_by_name=True)
        self.df = GBLibWrapper(None, self, self.gbp_iterator, True, True, True, ax_by_name=True)

        self.set_square()

        print (self.xnames)
        print (self.ynames)

    def ax(self, names, gname=None) :
        x, y = names
        xidx = self.xnames.index(x)
        yidx = self.ynames.index(y)

        idx = yidx * len(self.xnames) + xidx
        return self.ax_idx(idx, gname)

    def ax_idx(self, idx, gname=None) :
        gname = gname or self.used_gs
        return self.gsm[gname].get_make(idx)

    def clean_graph_labels(self, ** kwargs) :
        for ax in self.cgs :
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

        idx = len(self.xnames) * (len(self.ynames) - 1) - 1
        for name in self.xnames :
            idx += 1
            ax = self.ax_idx(idx)
            ax.set_xlabel(name, ** kwargs)

        for idx, name in enumerate(self.ynames) :
            idx = len(self.ynames) * idx
            ax = self.ax_idx(idx)
            ax.set_ylabel(name, ** kwargs)            

    @property
    def gbp_iterator(self) :
        return GroupByPlotter(self.data, self, (self.xname, self.yname)).iterator


