import os
import pylab as plt
import seaborn as sns

from seahorse import graphfun
from seahorse import constants

from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

class LibWrapper() :

    """
    Call a function from a library with arguments provided in bind 
    """

    def __init__(self, lib, ** binds) :
        self.lib =   lib
        self.binds = binds

    def __getattr__(self, funname) :
        if not hasattr(self.lib, funname) : 
            raise AttributeError("No function found : '%s'" %funname)
       
        return lambda * args, ** kwargs : getattr(self.lib, funname)(* args, ** self.binds, ** kwargs)

class GraphsGroup() :

    def __init__(self, axes, graph) :
        self.axes = axes
        self.graph = graph

    def __getattr__(self, name) :
        return lambda * args, ** kwargs : self.fun_wraps(name, * args, ** kwargs)

    def fun_wraps(self, name, * args, ** kwargs) :

        for ax in self.axes :
            obj = Graph(ax=ax) if self.graph else ax
            getattr(obj, name)(* args, ** kwargs)

class GBLibWrapper() :

    def __init__(self, lib, sc, iterator, bind_data=False, bind_axes=False, 
        data_call=False, order=None, title_prefix=None, colors=None, cuse=None) :
        
        self.lib = lib
        self.sc = sc
        self.iterator = iterator

        self.data_call = data_call
        self.bind_data = bind_data
        self.bind_axes = bind_axes
        
        self.order = order
        self.title_prefix = title_prefix
        self.colors = colors if colors is not None else sns.color_palette()
        self.cuse = cuse

    def fun_wrap(self, funname, * args, ** kwargs) :

        for idx, name, subdf in self.iterator :
            if subdf.empty : continue

            title = "%s %s" %(str(self.title_prefix), str(name)) if self.title_prefix else name
            idx = idx if not self.order else order.index(name)
            
            try : ax = self.sc.ax(idx)
            except IndexError : raise IndexError("Not enought axes available")
            
            if self.colors != False and "color" not in kwargs :
                try : color = self.colors[idx]
                except IndexError : color = self.colors
                kwargs["color"] = color

            if self.bind_data and self.cuse : kwargs["data"] = subdf[self.cuse]
            elif self.bind_data : kwargs["data"] = subdf
            if self.bind_axes : kwargs["ax"] = ax

            if constants.DEBUG :
                print (args)
                print (kwargs)

            self.get_fun(funname, kwargs)(* args, ** kwargs)
            ax.set_title(title)

        self.sc.clean_graph_labels()

    def __getattr__(self, funname) :
        return lambda * args, ** kwargs : self.fun_wrap(funname, * args, ** kwargs)
        
    def get_fun(self, funname, kwargs) :
        return self.get_fun_data(funname, kwargs) if self.data_call else self.get_fun_lib(funname)

    def get_fun_lib(self, funname) :
        return getattr(self.lib, funname)

    def get_fun_data(self, funname, kwargs) :
        data = kwargs.pop("data")
        return getattr(data, funname)

class GroupByPlotter() :

    def __init__(self, df, sc, hue, ignore_empty=True, ** kwargs) :

        self.df = df
        self.sc = sc
        self.hue = hue

        self.iterator = self.make_iterator(df, hue, ignore_empty)

        self.shs = GBLibWrapper(graphfun, sc, self.iterator, True, True, False, ** kwargs)
        self.sns = GBLibWrapper(sns, sc, self.iterator, True, True, False, ** kwargs)
        self.df = GBLibWrapper(None, sc, self.iterator, True, True, True, ** kwargs)

    def make_iterator(self, df, hue, ignore_empty) :

        if ignore_empty :
            iterator = ((name, sdf) for name, sdf in df.groupby(hue) if not sdf.empty)
        else :
            iterator = df.groupby(hue)

        for idx, group in enumerate(iterator) :
            name, subdf = group
            try : name = " - ".join(name) if not isinstance(name, str) else name
            except : name = str(name)
            yield idx, name, subdf

    def apply(self, fun, ** kwargs) :

        for idx, name, subdf in self.iterator :
            try : ax = self.sc.ax(idx)
            except IndexError : raise IndexError("Not enought axes available")
            fun(name, subdf, ax, ** kwargs)


class Fig() :
    # class to manage the mpl figure object
    # such as the figure size, etc

    def __init__(self, fig) :
        self.create_fig()

    def create_fig(self) :
        # Problem here is how pylab handles multiple figures
        # If plt.figure is used, sometimes (for example when using gridspec), you can have weird interaction
        # between different graphs
        # However if you use Figure() it works well but you cannot use the plt.show() function anymore

        if constants.SHOWMODE :
            self.fig = plt.figure()

        else :
            self.fig = Figure()
            self.canvas = constants.FCanva(self.fig)

        self.set_size_inches(* constants.DEFAULTRES)

    def set_square(self, size=12) :
        self.fig.set_size_inches(size, size)

    def set_size(self, pxwidth, pxheight, dpi=120) :
        self.fig.set_size_inches(pxwidth / float(dpi), pxheight / float(dpi))

    def set_size_inches(self, * args, ** kwargs) :
        self.fig.set_size_inches(* args, ** kwargs)

    def sfss(self) :
        self.fig.set_size_inches(20, 11.25)

    def tight_layout(self) :
        self.fig.tight_layout()

    def subplots_adjust(self, * args, ** kwargs) :
        self.fig.subplots_adjust(* args, ** kwargs)

    def gca(self, * args, ** kwargs) :
        self.fig.gca(*args, **kwargs)

    def show(self, * args, ** kwargs) :
        plt.show(self.fig, * args, ** kwargs)

    def save(self, fname, tab=True, ** kwargs) :
        kwargs = {"dpi":120, "alpha":0.5, ** kwargs}
        self.fig.savefig(fname, ** kwargs)

        if not tab : return
        fname = os.path.splitext(fname)[0] + ".tsv"
        self.data.to_csv(fname, sep="\t")


    def clear(self) :
        self.fig.clf()

    def close(self) :
        plt.close(self.fig)

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
    Labels, title and ticks
    """

    def set_labels(self, xlabel=None, ylabel=None, ** kwargs) :
        if xlabel is not None : self.ax.set_xlabel(str(xlabel), ** kwargs)
        if ylabel is not None : self.ax.set_ylabel(str(ylabel), ** kwargs)

    def transform_title(self, fun_names={}, ** kwargs) :
        fun = lambda x : fun_names.get(x, x) if isinstance(fun_names, dict) else fun_names
        title = fun(self.ax.get_title())
        self.ax.set_title(title, ** kwargs)

    def _ticks_names(self, ticks_iterator, fun_names) :
        for tick in ticks_iterator :
            text = tick.get_text()
            yield fun_names(text)

    def transform_xticks(self, fun_names={}, which="major", ** kwargs) :
        fun = (lambda x : fun_names.get(x, x)) if isinstance(fun_names, dict) else fun_names
        self.ax.set_xticklabels(list(self._ticks_names(self.ax.get_xticklabels(which=which), fun)), ** kwargs)

    def transform_yticks(self, fun_names={}, which="major", ** kwargs) :
        fun = (lambda x : fun_names.get(x, x)) if isinstance(fun_names, dict) else fun_names
        self.ax.set_yticklabels(list(self._ticks_names(self.ax.get_yticklabels(which=which), fun)), ** kwargs)

    def set_xticks(self, positions, values=None, * args, ** kwargs) :
        values = values or [str(position) for position in positions]
        self.ax.set_xticks(positions)
        self.ax.set_xticklabels(values, * args, ** kwargs)

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

        self.ax.set_xticklabels(labels, *args, **kwargs)

    """
    Legends
    """

    def remove_legend(self) :
        if self.ax.legend_ is None : return
        self.ax.legend_.remove()

    def set_legend(self, * args, ** kwargs) :
        self.ax.legend(*args, ** kwargs)

    def get_legend(self) :
        return self.ax.legend_

    def apply_legend(self, fun, * args, ** kwargs) :
        l = self.get_legend()
        if l is None : return
        getattr(l, fun)(* args, ** kwargs)

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

class SubplotsContainer(Fig) :
    
    def __init__(self, data, * args, sharex=False, sharey=False, ** kwargs) :
        self.data = data
        self._axes = {}
        self.gs = {}
        self.used_gs = "base"

        self.sharex = sharex
        self.sharey = sharey

        self.create_fig()
        self.add_gs("base", * args, ** kwargs)
        self.setup_ax_labels()

    @property
    def current_gs(self):
        return self.gs[self.used_gs]

    @property
    def current_gs_shape(self):
        return self.current_gs._nrows, self.current_gs._ncols

    @property
    def axes(self) :
        return self._axes.get(self.used_gs, [])

    def ax(self, idx) :
        try : return self.axes[idx]
        except IndexError : return self.get_ax(idx)
        except TypeError : return self.get_ax(idx)

    def graph(self, idx, data=None) :
        data = self.data if data is None else data
        return Graph(data, self.ax(idx))

    def get_axes_edge(self, top=False, bottom=False, left=False, right=False) :
        ncols = self.current_gs_shape[1]
        axes_list = []

        if top : axes_list += self.axes[:ncols]
        if bottom : axes_list += self.axes[- ncols:]
        if left : axes_list += self.axes[::ncols]
        if right : axes_list += self.axes[::-ncols][::-1]

        return axes_list

    def get_graph_top_right(self) :
        return Graph(self.data, self.get_axes_edge(top=True)[-1])

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

    def groupby(self, hue, ** kwargs) :
        return GroupByPlotter(self.data, self, hue, ** kwargs)

    def apply(self, graph=True) :
        return GraphsGroup(self.axes, graph)

    def add_subplot(self, ss) :
        kwargs = {}
        if self.sharex and self.axes : kwargs["sharex"] = self.axes[0]
        if self.sharey and self.axes : kwargs["sharey"] = self.axes[0]
        return self.fig.add_subplot(ss, ** kwargs)

    def get_ax(self, sterm) :
        ss = self.gs[self.used_gs].__getitem__(sterm)
        ax = self.add_subplot(ss)
        self._axes.setdefault(self.used_gs, []).append(ax)
        return ax

    def add_gs(self, name, * args, ** kwargs) :
        self.gs[name] = SubplotsContainer.get_gridspec(* args, ** kwargs)

    def get_graph(self, arg, df) :
        return Graph(data=df, ax=self[arg])

    @staticmethod
    def get_gridspec(* args, ** kwargs) :
        return gridspec.GridSpec(* args, ** kwargs)

    def gs_update(self, * args, gname="base", ** kwargs) :
        self.gs.update(* args, ** kwargs)

    def clean_graph_labels(self) :
        ax_left = self.get_axes_edge(left=True)
        ax_bottom = self.get_axes_edge(bottom=True)

        for ax in self.axes :
            if ax in ax_left and ax in ax_bottom : continue
            elif ax in ax_left : ax.set_xlabel("")
            elif ax in ax_bottom : ax.set_ylabel("")
            else : ax.set_ylabel(""); ax.set_xlabel("")

    def force_sharex(self) :
        ax_bottom = self.get_axes_edge(bottom=True)
        min_xlim = min(min(ax.get_xlim()) for ax in self.axes)
        max_xlim = max(max(ax.get_xlim()) for ax in self.axes)

        for ax in self.axes :
            ax.set_xlim((min_xlim, max_xlim))
            if ax not in ax_bottom : ax.set_xticklabels(["" for _ in ax.get_xticklabels()])

    def force_sharey(self) :
        ax_left = self.get_axes_edge(left=True)
        min_ylim = min(min(ax.get_ylim()) for ax in self.axes)
        max_ylim = max(max(ax.get_ylim()) for ax in self.axes)

        for ax in self.axes :
            ax.set_ylim((min_ylim, max_ylim))
            if ax not in ax_left : ax.set_yticklabels(["" for _ in ax.get_yticklabels()])

    def force_both(self) :
        self.force_sharex()
        self.force_sharey()

    def set_labels(self, xlabel=None, ylabel=None, sub=True, ** kwargs) :
        if sub : self._set_labels_sub(xlabel, ylabel, ** kwargs)
        else : self._set_labels_main(xlabel, ylabel, ** kwargs)

    def _set_labels_main(self, xlabel=None, ylabel=None, ** kwargs) :
        if xlabel : self.ax_labels.set_xlabel(xlabel, ** kwargs)
        if ylabel : self.ax_labels.set_xlabel(ylabel, ** kwargs)

    def _set_labels_sub(self, xlabel=None, ylabel=None, ** kwargs) :
        ax_left = self.get_axes_edge(left=True)
        ax_bottom = self.get_axes_edge(bottom=True)

        for ax in self.axes :
            ax_xlabel = xlabel if ax in ax_bottom else ""
            ax_ylabel = ylabel if ax in ax_left else ""

            ax.set_xlabel(ax_xlabel, ** kwargs)
            ax.set_ylabel(ax_ylabel, ** kwargs)

    def remove_xticks_top(self, which="major") :
        ax_bottom = self.get_axes_edge(bottom=True)
        fun = lambda x : ""

        for ax in self.axes :
            if ax not in ax_bottom :
                graph = Graph(ax=ax)
                graph.transform_xticks(fun, which=which)

    def select_legend(self, index, topright=True, ** kwargs) :

        if topright :
            kwargs.setdefault("loc", "upper left")
            kwargs.setdefault("bbox_to_anchor", (1, 1))
            kwargs.setdefault("prop", {"size" : 15})

        for idx, ax in enumerate(self.axes) :
            if idx == index :
                self.graph(idx).set_legend(** kwargs)
            else :
                self.graph(idx).remove_legend()