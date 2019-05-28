from collections import defaultdict, Counter
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

import pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import seahorse

try :
    from scipy.cluster.hierarchy import linkage, dendrogram
except ImportError :
    pass

class PyUpset(seahorse.Fig) :

    def __init__(self, df, key, value, unique=True, intersection=True, addvalue=True,
                 bcolor="black", griddots=False, lsize=12, maxwidth=10, row_clustering=False,
                 row_dendogram=False, method='average', metric='euclidean', 
                 boxratio=(None, None), spacers=(.1, .1), gridkwargs={}) :

        self.df = df 
        self.key = key
        self.value = value
        
        self.maxwidth = maxwidth
        self.addvalue = addvalue
        self.forced_names = None

        self.create_fig()
        self.init_ui(boxratio, spacers, row_dendogram)
        self.sim = self.get_sim(intersection, unique)
        
        if row_clustering or row_dendogram : self.run_clustering(method, metric)
        if row_dendogram : self.dendrogram()

        self.make_dist_barplot(bcolor=bcolor, lsize=lsize)
        self.make_comb_barplot(bcolor=bcolor, lsize=lsize)
        if griddots : self.make_grid_dots()
        else : self.make_grid_hm(gridkwargs)

    def init_ui(self, boxratio, spacers, row_dendogram) :
        # http://matplotlib.org/api/gridspec_api.html
        increment_idx_gs = 1
        default_width_ratios = [3, 9]
        default_height_ratios = (3, 1)

        if row_dendogram : 
            increment_idx_gs += 1
            default_width_ratios.insert(0, 3)

        width_ratios = boxratio[0] or tuple(default_width_ratios)
        height_ratios = boxratio[1] or default_height_ratios

        self.top_left_idx = increment_idx_gs
        self.gs = gridspec.GridSpec(2, increment_idx_gs + 1, width_ratios=width_ratios, height_ratios=height_ratios,
        hspace=spacers[0], wspace=spacers[1])

        self.ax_comb_barplot = self.fig.add_subplot(self.gs[increment_idx_gs])
        increment_idx_gs += 1
       
        if row_dendogram :
            self.ax_dendo = self.fig.add_subplot(self.gs[increment_idx_gs])
            increment_idx_gs += 1

        self.ax_dist_barplot = self.fig.add_subplot(self.gs[increment_idx_gs])
        increment_idx_gs += 1

        self.ax_grid = self.fig.add_subplot(self.gs[increment_idx_gs])

    def get_top_left(self) :
        return self.fig.add_subplot(self.gs[:self.top_left_idx])

    """
    Utilities
    """

    @property
    def unique(self) :
        return sorted(self.df[self.value].unique())

    @property
    def names(self):
        if self.forced_names : return self.forced_names
        return sorted({name for names, count in self.sim.most_common(self.maxwidth)
            for name in names})
    
    @property
    def ncombi(self) :
        n = self.df[self.value].nunique()
        return sum(factorial(n) / (factorial(r) * factorial(n - r))
        for r in range(1, n + 1))

    def get_sim(self, intersection, unique) :
        return PyUpset.get_sim_df(self.df, self.key, self.value, intersection, unique)

    @staticmethod
    def get_sim_df(df, key, value, intersection=True, unique=True) :
        # just a static method to make outside call without producing upset plot
        
        counter = Counter(tuple(sorted(sdf[value].unique())) for key, sdf in df.groupby(key))

        if not intersection :
            ncounter = Counter()
            for keys, value in counter.items() :
                ncounter[keys] += value
                for subkeys in PyUpset.all_combinations(keys) :
                    subkeys = tuple(sorted(subkeys))
                    ncounter[subkeys] += 1
            counter = ncounter

        if not unique :
            counter = Counter({key : value for key, value in counter.items() if len(key) != 1})

        return counter

    @staticmethod
    def all_combinations(iterable) :
        return (e for i in range(1, len(iterable) + 1)
        for e in combinations(iterable, i) if e != iterable)

    @staticmethod
    def as_matrix(df, key, value, default_value=0, names=True) :
        if names : df = df[df[value].isin(names)]
        df["__value__"] = 1
        return pd.pivot_table(df, index=key, columns=value, values="__value__", fill_value=default_value)

    """
    Clustering
    """

    def run_clustering(self, method, metric) :
        df = PyUpset.as_matrix(self.df, self.key, self.value, names=self.names)
        self.dend = linkage(df.T.as_matrix(), method, metric)
        result = dendrogram(self.dend, no_plot=True)
        self.forced_names = [df.columns[idx] for idx in result["leaves"]]

    def dendrogram(self) :
        dendrogram(self.dend, orientation="left", ax=self.ax_dendo, link_color_func=lambda x : "black")
        plt.setp(self.ax_dendo.get_xticklabels(), visible=False)
        plt.setp(self.ax_dendo.get_yticklabels(), visible=False)
        self.ax_dendo.set_facecolor('white')
        self.ax_dendo.axis("off")

    """
    Basic plots
    """

    def make_dist_barplot(self, bcolor="black", lsize=15) :
        df = self.df[self.df[self.value].isin(self.names)]
        df = df.groupby(self.value).size()
        df = df.reindex(self.names)

        df.plot.barh(ax=self.ax_dist_barplot, color=bcolor)
        self.ax_dist_barplot.set_facecolor('white')
        self.ax_dist_barplot.invert_xaxis()
        self.ax_dist_barplot.set_xlabel("Set size", size=lsize)
        self.ax_dist_barplot.set_ylabel("")
        self.ax_dist_barplot.yaxis.tick_right()

        self.ax_dist_barplot.spines['left'].set_visible(False)
        self.ax_dist_barplot.spines['top'].set_visible(False)

    def make_comb_barplot(self, bcolor="black", lsize=15) :
        values = {}
        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sim_values = self.sim.most_common(int(fill))
        for idx in range(int(fill)) :
            try : values[idx] = sim_values[idx][1]
            except IndexError : values[idx] = 0

        s = pd.Series(values)
        s.plot.bar(ax=self.ax_comb_barplot, color=bcolor)
        self.ax_comb_barplot.set_facecolor('white')
        plt.setp(self.ax_comb_barplot.get_xticklabels(), visible=False)
        self.ax_comb_barplot.set_ylabel("Intersection size", size=lsize)
        if self.addvalue : PyUpset.barplot_add_value(self.ax_comb_barplot)

        self.ax_comb_barplot.spines['right'].set_visible(False)
        self.ax_comb_barplot.spines['top'].set_visible(False)

    @staticmethod
    def barplot_add_value(ax) :
        spacer = ax.get_ylim()[1] * 2/100.
        sum_v = float(sum([p.get_height() for p in ax.patches]))

        for p in ax.patches:
            height = p.get_height()
            height_name = int(height)
            x = p.get_x() + p.get_width() / 2.
            ax.text(x, height + spacer, str(height_name), ha="center", va="bottom", rotation=90)

    """
    Combinations plots
    """

    def make_grid_dots(self, dcolor="grey", tcolor="black", lwidth=3, msize=8) :
        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sorted_combinations = [c[0] for c in self.sim.most_common(int(fill))]
        width = len(sorted_combinations) if len(sorted_combinations) > fill else fill

        # Grey dots
        values = [(i + .5, j + .5) for i in range(int(width)) for j in range(len(self.names))]
        xvalues, yvalues = zip(*values)
        self.ax_grid.plot(xvalues, yvalues, 'o', markersize=msize, color=dcolor)

        # Combinations colors
        columns = self.names
        for idx, combination in enumerate(sorted_combinations) :
            dots = [(idx + .5, columns.index(column) + .5) for column in combination]
            xvalues, yvalues = zip(*dots)
            self.ax_grid.plot(xvalues, yvalues, 'o-', markersize=msize, linewidth=lwidth, color=tcolor)

        self.ax_grid.set_xlim((0, min(self.ncombi, self.maxwidth)))
        self.ax_grid.set_ylim((0, len(self.names)))
        plt.setp(self.ax_grid.get_xticklabels(), visible=False)
        plt.setp(self.ax_grid.get_yticklabels(), visible=False)
        self.ax_grid.axis("off")

    def make_grid_hm(self, gridkwargs) :
        # draw the grid as a heatmap

        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sorted_combinations = [c[0] for c in self.sim.most_common(int(fill))]

        hm_data = [{column : 1 if column in combination else 0 for column in self.names}
                   for idx, combination in enumerate(sorted_combinations)]

        df = pd.DataFrame(hm_data).T
        df = df.reindex(self.names[::-1])

        sns.heatmap(data=df, ax=self.ax_grid, xticklabels=False, yticklabels=False, cbar=False, linewidths=1,
        cmap="Greys", ** gridkwargs)

    """
    Supplemental draws
    """

    def draw_intersize_count(self, ax=None, ** kwargs) :
        count = self.df.groupby(self.key)[self.value].nunique().rename("intersize").reset_index()
        count = count.groupby("intersize").size().rename("count").reset_index()

        graph = seahorse.Graph(count, ax=ax)
        graph.sns.barplot(x="intersize", y="count", ** kwargs)
        return graph

class PyUpsetDic(PyUpset) :

    def __init__(self, dic, addvalue=True, bcolor="black", griddots=False, lsize=12, 
                 maxwidth=10, row_clustering=False, row_dendogram=False, 
                 method='average', metric='euclidean', boxratio=(None, None), 
                 spacers=(.1, .1), gridkwargs={}) :

        self.sim = Counter(dic)      
        self.maxwidth = maxwidth
        self.addvalue = addvalue
        self.forced_names = None

        self.create_fig()
        self.init_ui(boxratio, spacers, row_dendogram)
        
        if row_clustering or row_dendogram : self.run_clustering(method, metric)
        if row_dendogram : self.dendrogram()

        self.make_dist_barplot(bcolor=bcolor, lsize=lsize)
        self.make_comb_barplot(bcolor=bcolor, lsize=lsize)
        if griddots : self.make_grid_dots()
        else : self.make_grid_hm(gridkwargs)

    @property
    def ncombi(self) :
        n = len(self.sim)

        if n > 10 and not self.maxwidth :
            raise ValueError("More than 1000 combinations possible, please set a maxwidth")

        if n > 10 :
            return np.inf

        return sum(factorial(n) / (factorial(r) * factorial(n - r))
        for r in range(1, n + 1))

    def make_dist_barplot(self, bcolor="black", lsize=15) :


        mc = dict(self.sim.most_common(self.maxwidth))
        data = defaultdict(int)
        
        for keys, value in mc.items() :
            for key in keys :
                data[key] += value
   
        df = pd.Series(data).reindex(self.names)

        df.plot.barh(ax=self.ax_dist_barplot, color=bcolor)
        self.ax_dist_barplot.set_facecolor('white')
        self.ax_dist_barplot.invert_xaxis()
        self.ax_dist_barplot.set_xlabel("Set size", size=lsize)
        self.ax_dist_barplot.set_ylabel("")
        self.ax_dist_barplot.yaxis.tick_right()

        self.ax_dist_barplot.spines['left'].set_visible(False)
        self.ax_dist_barplot.spines['top'].set_visible(False)

    def make_comb_barplot(self, bcolor="black", lsize=15) :
        values = {}
        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sim_values = self.sim.most_common(int(fill))
        for idx in range(int(fill)) :
            try : values[idx] = sim_values[idx][1]
            except IndexError : values[idx] = 0

        s = pd.Series(values)
        s.plot.bar(ax=self.ax_comb_barplot, color=bcolor)
        self.ax_comb_barplot.set_facecolor('white')
        plt.setp(self.ax_comb_barplot.get_xticklabels(), visible=False)
        self.ax_comb_barplot.set_ylabel("Intersection size", size=lsize)
        if self.addvalue : PyUpset.barplot_add_value(self.ax_comb_barplot)

        self.ax_comb_barplot.spines['right'].set_visible(False)
        self.ax_comb_barplot.spines['top'].set_visible(False)

class PyUpsetHue(PyUpset) :

    def __init__(self, df, key, value, hue, nvalue=0, unique=True, intersection=True,
                 griddots=False, lsize=12, maxwidth=10, palette=None, addvalue=True,
                 boxratio=(None, None), spacers=(.1, .1), gridkwargs={}, legend=True) :

        self.df = df 
        self.key = key
        self.value = value
        self.hue = hue

        self.maxwidth = maxwidth
        self.addvalue = addvalue
        self.legend = legend
        self.palette = palette or seahorse.color_palette()

        self.create_fig()
        self.init_ui(boxratio, spacers, False)
        self.sim = self.get_sim(intersection, unique)
        
        self.make_dist_barplot(nvalue, lsize=lsize)
        self.make_comb_barplot(lsize=lsize)
        if griddots : self.make_grid_dots()
        else : self.make_grid_hm(gridkwargs)

    """
    Basic data
    """

    def get_sim(self, intersection, unique) :
        return {hue : PyUpset.get_sim_df(df, self.key, self.value, intersection, unique)
            for hue, df in self.df.groupby(self.hue)}

    def get_most_common(self, fill) :
        sim_values = [(name, size, hue) for hue, sim in self.sim.items()
            for name, size in sim.most_common(int(fill))]

        return sorted(sim_values, key = lambda x : x[1], reverse=True)[:int(fill)]

    @property
    def names(self):
        return sorted({name for names, count, hue in self.get_most_common(self.maxwidth)
            for name in names}) 

    """
    Basic plots
    """

    def make_dist_barplot(self, nvalue, lsize=15) :

        df = self.df.groupby([self.value, self.hue]).size().rename("__count__").reset_index()
        df = df[df[self.value].isin(self.names)]
        df = pd.pivot_table(df, index=self.value, columns=self.hue, values="__count__")
        df.reindex(self.names)

        df.plot.barh(ax=self.ax_dist_barplot, legend=False)

        self.ax_dist_barplot.set_facecolor('white')
        self.ax_dist_barplot.invert_xaxis()
        self.ax_dist_barplot.set_xlabel("Set size", size=lsize)
        self.ax_dist_barplot.set_ylabel("")
        self.ax_dist_barplot.yaxis.tick_right()

        self.ax_dist_barplot.spines['left'].set_visible(False)
        self.ax_dist_barplot.spines['top'].set_visible(False)

    def make_comb_barplot(self, lsize=15) :

        values = {}
        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sim_values = self.get_most_common(fill)    
        hues = sorted(self.df[self.hue].unique())
        colors = []

        for idx in range(int(fill)) :
            try : 
                values[idx] = sim_values[idx][1]
                colors.append(self.palette[hues.index(sim_values[idx][2])])
            
            except IndexError : 
                values[idx] = 0

        s = pd.Series(values)
        s.plot.bar(ax=self.ax_comb_barplot, color=colors, legend=False)
        self.ax_comb_barplot.set_facecolor('white')
        plt.setp(self.ax_comb_barplot.get_xticklabels(), visible=False)
        self.ax_comb_barplot.set_ylabel("Intersection size", size=lsize)
        if self.addvalue : PyUpset.barplot_add_value(self.ax_comb_barplot)

        # add legend
        if self.legend :
            colors = {hue : self.palette[idx] for idx, hue in enumerate(sorted(hues))}
            seahorse.basic_legend(self.ax_comb_barplot, colors)

        self.ax_comb_barplot.spines['right'].set_visible(False)
        self.ax_comb_barplot.spines['top'].set_visible(False)

    """
    Combinations plots
    """

    def make_grid_dots(self, dcolor="grey", tcolor="black", lwidth=3) :
        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sorted_combinations = [c[0] for c in self.get_most_common(int(fill))]
        width = len(sorted_combinations) if len(sorted_combinations) > fill else fill

        # Grey dots
        values = [(i + .5, j + .5) for i in range(int(width)) for j in range(len(self.names))]
        xvalues, yvalues = zip(*values)
        self.ax_grid.plot(xvalues, yvalues, 'o', markersize=12, color=dcolor)

        # Combinations colors
        for idx, combination in enumerate(sorted_combinations) :
            dots = [(idx + .5, self.names.index(column) + .5) for column in combination]
            xvalues, yvalues = zip(*dots)
            self.ax_grid.plot(xvalues, yvalues, 'o-', markersize=12, linewidth=lwidth, color=tcolor)

        self.ax_grid.set_xlim((0, min(self.ncombi, self.maxwidth)))
        self.ax_grid.set_ylim((0, len(self.names)))
        plt.setp(self.ax_grid.get_xticklabels(), visible=False)
        plt.setp(self.ax_grid.get_yticklabels(), visible=False)
        self.ax_grid.axis("off")

    def make_grid_hm(self, gridkwargs) :
        # draw the grid as a heatmap

        fill = self.ncombi if self.ncombi < self.maxwidth else self.maxwidth
        sorted_combinations = [c[0] for c in self.get_most_common(int(fill))]

        hm_data = []
        columns = self.names
        for idx, combination in enumerate(sorted_combinations) :
            hm_data.append({column : 1 if column in combination else 0 for column in columns})

        df = pd.DataFrame(hm_data).T
        df = df.reindex(columns[::-1])

        for i in range(idx+1, int(fill)) :
            df[i] = 0

        sns.heatmap(data=df, ax=self.ax_grid, xticklabels=False, yticklabels=False, cbar=False, linewidths=1,
        cmap="Greys", ** gridkwargs)