import os, copy
import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from seahorse import graphfun
from seahorse import constants

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator, LogLocator
from matplotlib import rcParams

class SHException(Exception) :
    pass

class LibWrapper() :

    """
    Call a function from a library with arguments provided in bind 
    """

    def __init__(self, lib, ** binds) :
        self.lib = lib
        self.binds = {key : value for key, value in binds.items() if value is not None}

    def launch_fun(self, funname, * args, ** kwargs) :
        binds = copy.copy(self.binds)
        
        if "cuse" in kwargs :
            cuse = kwargs.pop("cuse", None)
            if cuse and "data" in binds : binds["data"] = binds["data"][cuse]
            if not cuse : binds.pop("data", None)

        return getattr(self.lib, funname)(* args, ** binds, ** kwargs)

    def __getattr__(self, funname) :
        if not hasattr(self.lib, funname) : 
            if constants.DEBUG : print (self.lib, funname)
            raise AttributeError("No function found : '%s'" %(funname))

        return lambda * args, ** kwargs : self.launch_fun(funname, * args, ** kwargs)

class GraphAttributes(list) :

    def __init__(self, elements, graph) :
        self.graph = graph
        super(GraphAttributes, self).__init__(elements)

    def __getattr__(self, name) :

        new_elements = []

        for element in self :
            obj = getattr(Graph(ax=element), name) if self.graph else getattr(element, name)
            new_elements.append(obj)

        return GraphAttributes(new_elements, False)

    def __call__(self, * args, ** kwargs) :
        return [element(* args, ** kwargs) for element in self]

class GBLibWrapper() :

    def __init__(self, lib, sc, iterator, bind_data=False, bind_axes=False, 
        data_call=False, order=None, title_prefix=None, colors=None, itcolors=True, cuse=None) :
        
        self.lib = lib
        self.sc = sc
        self.iterator = iterator

        self.data_call = data_call
        self.bind_data = bind_data
        self.bind_axes = bind_axes
        
        self.order = order
        self.title_prefix = title_prefix
        self.colors = colors if colors is not None else sns.color_palette()
        self.itcolors = itcolors
        self.cuse = cuse

    def fun_wrap(self, funname, * args, ** kwargs) :

        for idx, name, subdf in self.iterator :
            if subdf.empty : continue

            title = "%s %s" %(str(self.title_prefix), str(name)) if self.title_prefix else name
            idx = idx if not self.order else order.index(name)
            
            try : ax = self.sc.ax(idx)
            except IndexError : raise IndexError("Not enought axes available")
            
            if self.itcolors and self.colors != False and "color" not in kwargs :
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

    def __init__(self, df, sc, hue, ignore_empty=True,
        clean_cat=False, ** kwargs) :

        self.df = df
        self.sc = sc
        self.hue = hue

        self.iterator = self.make_iterator(df, hue, ignore_empty, clean_cat)

        self.shs = GBLibWrapper(graphfun, sc, self.iterator, True, True, False, ** kwargs)
        self.sns = GBLibWrapper(sns, sc, self.iterator, True, True, False, ** kwargs)
        self.df = GBLibWrapper(None, sc, self.iterator, True, True, True, ** kwargs)

    @staticmethod
    def clean_cat(subdf) :
        
        for column in subdf.select_dtypes(include=['category']) :
            found_values = set(subdf[column].unique())
            used_categories = [cat for cat in subdf[column].cat.categories if cat in found_values]
            subdf[column] = pd.Categorical(subdf[column], categories=used_categories)

        return subdf

    def make_iterator(self, df, hue, ignore_empty, clean_cat) :

        if ignore_empty :
            iterator = ((name, sdf) for name, sdf in df.groupby(hue) if not sdf.empty)
        
        else :
            iterator = df.groupby(hue)

        for idx, group in enumerate(iterator) :
            name, subdf = group
            if clean_cat : subdf = GroupByPlotter.clean_cat(subdf)
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
        return self.fig

    def get_size(self, px=False) :
        size = self.fig.get_size_inches()
        if px : size = size * self.fig.dpi
        return size

    def set_square(self, size=12) :
        self.fig.set_size_inches(size, size)

    def set_size(self, pxwidth, pxheight, dpi=120) :
        self.fig.set_size_inches(pxwidth / float(dpi), pxheight / float(dpi))

    def set_size_inches(self, * args, ** kwargs) :
        self.fig.set_size_inches(* args, ** kwargs)

    def sfss(self) :
        self.fig.set_size_inches(20, 11.25)

    def rotate_fig(self) :
        size = self.fig.get_size_inches()
        self.set_size_inches(* size[::-1])

    def tight_layout(self) :
        self.fig.tight_layout()

    def subplots_adjust(self, * args, ** kwargs) :
        self.fig.subplots_adjust(* args, ** kwargs)

    def gca(self, * args, ** kwargs) :
        self.fig.gca(*args, ** kwargs)

    def show(self, * args, ** kwargs) :
        plt.show(self.fig, * args, ** kwargs)

    def save(self, fname, tab=True, ** kwargs) :
        kwargs = {"dpi": 120, "alpha": 0.5, ** kwargs}

        self.fig.savefig(fname, ** kwargs)

        if not tab : return
        fname = os.path.splitext(fname)[0] + ".tsv"
        
        # for some graph object (such as PyUpset) we don't store data
        try : self.data.to_csv(fname, sep="\t")
        except AttributeError : pass

    @staticmethod
    def save_pdfs(fname, figs, * args, ** kwargs) :
        kwargs = {"dpi":120, "alpha":0.5, ** kwargs}
        with PdfPages(fname) as pdf :
            for fig in figs :
                pdf.savefig(fig.fig, * args, ** kwargs)

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

    def set_xlabel(self, xlabel=None, ** kwargs) :
        if kwargs and not xlabel : xlabel = self.ax.get_xlabel()
        self.ax.set_xlabel(xlabel, ** kwargs)

    def set_ylabel(self, ylabel=None, ** kwargs) :
        if kwargs and not ylabel : ylabel = self.ax.get_ylabel()
        self.ax.set_ylabel(ylabel, ** kwargs)        

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

class MovingPosition() :

    def __init__(self, axe_x, axe_y, angle, distance, fig, ax) :
        self.fig = fig
        self.ax = ax
        
        self.axe_x = axe_x
        self.axe_y = axe_y

        self.distance = distance
        self.angle = angle

    def __float__(self) :

        print ("-" * 10)
        print (self.fig.get_size_inches())
        print (self.ax.get_position().bounds)

        return float(0 - self.distance)


class BrokenGraph(Fig) :

    def __init__(self, xlims=None, ylims=None, data=None, copy=False, fig=None, gskwargs={}, 
                 despine=True, d=.01, tilt=30, ** kwargs) :

        # Highly inspired by this repo
        # https://github.com/bendichter/brokenaxes/blob/master/brokenaxes.py

        # see also
        # https://matplotlib.org/examples/pylab_examples/broken_axis.html

        # Short explanation of how it works
        # For each part, we make a different subplot
        # Each subplot corresponds to a xlim or ylim
        # We then make it looks like a unique ax using by removing the background
        # And adding line on axis

        fig = fig or self.create_fig()

        width_ratios = [i[1] - i[0] for i in xlims] if xlims else [1]
        height_ratios = [i[1] - i[0] for i in ylims[::-1]] if ylims else [1]

        ncols, nrows = len(width_ratios), len(height_ratios)

        gskwargs.update(ncols=ncols, nrows=nrows, height_ratios=height_ratios,
                        width_ratios=width_ratios)

        self.gs = gridspec.GridSpec(** gskwargs)
        self.ax = plt.Subplot(fig, gridspec.GridSpec(1, 1)[0])

        self.axes = []
        for igs in self.gs :
            subax = plt.Subplot(fig, igs)
            fig.add_subplot(subax)
            self.axes.append(subax)

        fig.add_subplot(self.ax)

        for i, subax in enumerate(self.axes) :
            if ylims is not None :
                subax.set_ylim(ylims[::-1][i // ncols])
            if xlims is not None :
                subax.set_xlim(xlims[i % ncols])
                subax.spines['left'].set_visible(False)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.patch.set_facecolor('none')

        if d :
            self.draw_diags(d, tilt, despine)

        if despine :
            self.set_spines()

    @staticmethod
    def test() :
        # For now it doesn't work
        # I'm not able to show the spine correctly

        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        #graph = Graph()
        graph = BrokenGraph(xlims=((0, .75), (1.5, 3)), d=.01, tilt=60)

        graph.graphs.ax.plot(t, s)
        graph.tight_layout()
        return graph

    @property
    def graphs(self, * args, ** kwargs) :
        return GraphAttributes(self.axes, True)


    @staticmethod
    def draw_diag(ax, xpos, xlen, ypos, ylen, ** kwargs):
        return ax.plot((xpos - xlen, xpos + xlen), (ypos - ylen, ypos + ylen),
            ** kwargs)

    def draw_diags(self, d, tilt, despine):
        """
        
        Parameters
        ----------
        d: float
            Length of diagonal split mark used to indicate broken axes
        tilt: float
            Angle of diagonal split mark
        """

        d_kwargs = dict(color="black", clip_on=False)
        
        ds = []
        for ax in self.axes :
            d_kwargs["transform"] = ax.transAxes

            bounds = ax.get_position().bounds
            print (ax.transAxes.transform((0, 0)))

            if ax.is_last_row() :

                # draw on x axis

                if not ax.is_first_col() :

                    x0 = MovingPosition(0, 0, tilt, -.01, self.fig, ax)
                    x1 = MovingPosition(0, 0, tilt, -.01, self.fig, ax)

                    y0 = MovingPosition(0, 0, tilt, +.01, self.fig, ax)
                    y1 = MovingPosition(0, 0, tilt, +.01, self.fig, ax)

                    ax.plot((x0, y0), (x1, y1), ** d_kwargs)

                if not ax.is_last_col() :
                    ax.plot((1 + d, 1 - d), (d, -d), ** d_kwargs)

            if ax.is_first_col() :
                
                # draw on y axis

                if not ax.is_first_row() :
                    ax.plot((d, -d), (d, -d), ** d_kwargs)
                
                if not ax.is_last_row() :
                    ax.plot((d, -d), (1 + d, 1 - d), ** d_kwargs)

        self.diag_handles = ds

    def set_spines(self):
        """Removes the spines of internal axes that are not boarder spines.
        """
        for ax in self.axes :
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            if not ax.is_last_row() :
                ax.spines['bottom'].set_visible(False)
                plt.setp(ax.xaxis.get_minorticklabels(), visible=False)
                plt.setp(ax.xaxis.get_minorticklines(), visible=False)
                plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
                plt.setp(ax.xaxis.get_majorticklines(), visible=False)
            
            if  not ax.is_first_row() :
                ax.spines['top'].set_visible(False)
            
            if not ax.is_first_col() :
                ax.spines['left'].set_visible(False)
                plt.setp(ax.yaxis.get_minorticklabels(), visible=False)
                plt.setp(ax.yaxis.get_minorticklines(), visible=False)
                plt.setp(ax.yaxis.get_majorticklabels(), visible=False)
                plt.setp(ax.yaxis.get_majorticklines(), visible=False)
            
            if not ax.is_last_col() :
                ax.spines['right'].set_visible(False)

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
        return self._axes.get(self.used_gs, {})

    @property
    def faxes(self):
        # flatten and filled ax with None
        if not self.axes : return []
        return [self.axes.get(idx, None) for idx in range(max(self.axes) + 1)]
    
    @property
    def nffaxes(self):
        # non filled flatten ax
        return [ax for ax in self.faxes if ax is not None]
    
    def fill_axes(self) :
        nrow, ncol = self.current_gs_shape
        return [self.ax(i) for i in range(nrow * ncol)]


    def ax(self, idx) :
        try : return self.axes[idx]
        except KeyError : return self.get_ax(idx)

    def graph(self, idx, data=None) :
        data = self.data if data is None else data
        return Graph(data, self.ax(idx))

    def get_axes_edge(self, top=False, bottom=False, left=False, right=False) :
        ncols = self.current_gs_shape[1]
        axes_list = []
        faxes = self.faxes

        if top : axes_list += faxes[:ncols]
        if bottom : axes_list += faxes[- ncols:]
        if left : axes_list += faxes[::ncols]
        if right : axes_list += faxes[::-ncols][::-1]

        return [ax for ax in axes_list if ax is not None]

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
        return GraphAttributes(self.nffaxes, graph)

    def add_subplot(self, ss) :
        kwargs = {}
        if self.sharex and self.nffaxes : kwargs["sharex"] = self.nffaxes[0]
        if self.sharey and self.nffaxes : kwargs["sharey"] = self.nffaxes[0]
        return self.fig.add_subplot(ss, ** kwargs)

    def get_ax(self, sterm) :
        try : ss = self.gs[self.used_gs].__getitem__(sterm)
        except IndexError : raise SHException("Axes index doesn't exist. You subplot properties might be wrong.")
        ax = self.add_subplot(ss)
        self._axes.setdefault(self.used_gs, {})[sterm] = ax
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

        for ax in self.nffaxes :
            if ax in ax_left and ax in ax_bottom : continue
            elif ax in ax_left : ax.set_xlabel("")
            elif ax in ax_bottom : ax.set_ylabel("")
            else : ax.set_ylabel(""); ax.set_xlabel("")

    def clean_xticks_bottom(self) :
        ax_bottom = self.get_axes_edge(bottom=True)
        for ax in self.nffaxes :
            if ax not in ax_bottom : 
                ax.set_xticklabels(["" for _ in ax.get_xticklabels()])

    def force_sharex(self) :
        min_xlim = min(min(ax.get_xlim()) for ax in self.nffaxes)
        max_xlim = max(max(ax.get_xlim()) for ax in self.nffaxes)
        for ax in self.nffaxes : ax.set_xlim((min_xlim, max_xlim))
        self.clean_xticks_bottom()        

    def force_sharey(self) :
        ax_left = self.get_axes_edge(left=True)
        min_ylim = min(min(ax.get_ylim()) for ax in self.nffaxes)
        max_ylim = max(max(ax.get_ylim()) for ax in self.nffaxes)

        for ax in self.nffaxes :
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
        if ylabel : self.ax_labels.set_ylabel(ylabel, ** kwargs)

    def _set_labels_sub(self, xlabel=None, ylabel=None, ** kwargs) :
        ax_left = self.get_axes_edge(left=True)
        ax_bottom = self.get_axes_edge(bottom=True)

        for ax in self.nffaxes :
            ax_xlabel = xlabel if ax in ax_bottom else ""
            ax_ylabel = ylabel if ax in ax_left else ""

            ax.set_xlabel(ax_xlabel, ** kwargs)
            ax.set_ylabel(ax_ylabel, ** kwargs)

    def remove_xticks_top(self, which="major") :
        ax_bottom = self.get_axes_edge(bottom=True)
        fun = lambda x : ""

        for ax in self.nffaxes :
            if ax not in ax_bottom :
                graph = Graph(ax=ax)
                graph.transform_xticks(fun, which=which)

    def select_legend(self, index, topright=True, ** kwargs) :

        if topright :
            kwargs.setdefault("loc", "upper left")
            kwargs.setdefault("bbox_to_anchor", (1, 1))
            kwargs.setdefault("prop", {"size" : 15})

        for idx, ax in enumerate(self.nffaxes) :
            if idx == index :
                self.graph(idx).set_legend(** kwargs)
            else :
                self.graph(idx).remove_legend()

