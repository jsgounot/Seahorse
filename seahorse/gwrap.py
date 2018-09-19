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

class GBLibWrapper() :

    def __init__(self, lib, sc, iterator, bind_data=False, bind_axes=False, 
        data_call=False, order=None, title_prefix=None, colors=None) :
        
        self.lib = lib
        self.sc = sc
        self.iterator = iterator

        self.data_call = data_call
        self.bind_data = bind_data
        self.bind_axes = bind_axes
        
        self.order = order
        self.title_prefix = title_prefix
        self.colors = colors if colors is not None else sns.color_palette()

    def fun_wrap(self, funname, * args, ** kwargs) :

        for idx, name, subdf in self.iterator :
            if subdf.empty : continue

            title = "%s %s" %(str(self.title_prefix), str(name)) if self.title_prefix else name
            idx = idx if not self.order else order.index(name)
            
            try : ax = self.sc.ax(idx)
            except IndexError : raise IndexError("Not enought axes available")
            
            if self.colors != False :
                try : color = self.colors[idx]
                except IndexError : color = self.colors
                kwargs["color"] = color

            if self.bind_data : kwargs["data"] = subdf
            if self.bind_axes : kwargs["ax"] = ax

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
        iterator = self.make_iterator(df, hue, ignore_empty)

        self.shs = GBLibWrapper(graphfun, sc, iterator, True, True, False, ** kwargs)
        self.sns = GBLibWrapper(sns, sc, iterator, True, True, False, ** kwargs)
        self.df = GBLibWrapper(None, sc, iterator, True, True, True, ** kwargs)

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

    def save(self, fname, ** kwargs) :
        kwargs = {"dpi":120, "alpha":0.5, ** kwargs}
        self.fig.savefig(fname, ** kwargs)

    def clear(self) :
        self.fig.clf()

    def close(self) :
        plt.close(self.fig)

class Graph(Fig) :

    def __init__(self, data=None, ax=None, copy=False, ** kwargs) :
        self.data = copy(data) if copy else data

        if ax is None :
            self.create_fig()
            self.ax = self.fig.add_subplot(111, ** kwargs)
        
        else :
            self.ax =  ax
            self.fig = ax.get_figure()

        self.df =  LibWrapper(self.data, ax = self.ax)
        self.sns = LibWrapper(sns, data = self.data, ax = self.ax)
        self.shs = LibWrapper(graphfun, data = self.data, ax = self.ax)

    """
    Labels, title and ticks
    """

    def set_labels(self, xlabel=None, ylabel=None, ** kwargs) :
        if xlabel is not None : self.ax.set_xlabel(str(xlabel), ** kwargs)
        if ylabel is not None : self.ax.set_ylabel(str(ylabel), ** kwargs)

    def _ticks_names(self, ticks_iterator, new_names) :
        for tick in ticks_iterator :
            text = tick.get_text()
            yield new_names.get(text, text)

    def transform_xticks(self, new_names={}, which="major", ** kwargs) :
        self.ax.set_xticklabels(list(self._ticks_names(self.ax.get_xticklabels(which=which), new_names)), ** kwargs)

    def transform_yticks(self, new_names={}, which="major", ** kwargs) :
        self.ax.set_yticklabels(list(self._ticks_names(self.ax.get_yticklabels(which=which), new_names)), ** kwargs)

    """
    Legends
    """

    def remove_legend(self) :
        if self.ax.legend_ is None : return
        self.ax.legend_.remove()

    def set_legend(self, *args, **kwargs) :
        self.ax.legend(*args, **kwargs)

    def get_legend(self) :
        return self.ax.legend_

    def apply_legend(self, fun, * args, ** kwargs) :
        l = self.get_legend()
        if l is None : return
        getattr(l, fun)(*args, **kwargs)

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
        try : self.axes[idx]
        except IndexError : return self.get_ax(idx)
        except TypeError : return self.get_ax(idx)

    def graph(self, idx) :
        return Graph(self.data, self.ax(idx))

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
