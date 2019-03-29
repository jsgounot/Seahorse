# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-12-21 10:49:13
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 16:10:28

import numpy as np
from seahorse import Graph, Fig, sns
import matplotlib.gridspec as gridspec

class ScatterDist(Fig) :
    def __init__(self, df, c1, c2, hue=None, kwargs_scatter={}, kwargs_dist={}, method="violinplot",
                 spacers=(.1, .1), boxratio=[(3,1), (1,3)], size=12, fillna=None, palette=None,
                 xlim=None, ylim=None) :
        
        # Maybe rebuild this using that :
        # https://github.com/mwaskom/seaborn/issues/1194

        self.df = df

        if method not in ["violinplot", "boxplot"] :
            raise ValueError("method should be either distplot, boxplot or violinplot")

        self.clean_na(c1, c2, fillna)
        palette = sns.color_palette() if palette is None else palette
        if hue is not None : 
            kwargs_dist, kwargs_scatter = self.update_hue(hue, kwargs_scatter, kwargs_dist, palette)

        self.create_fig()
        self.init_ui(boxratio, spacers, size)
        self.draw_scatter(c1, c2, xlim, ylim, ** kwargs_scatter)
        self.draw_dists(c1, c2, method, ** kwargs_dist)

        self.data = df

    def update_hue(self, hue, kwargs_dist, kwargs_scatter, palette) :
        kwargs_scatter["hue"] = hue
        kwargs_scatter["palette"] = palette
        kwargs_dist["hue"] = hue
        kwargs_dist["palette"] = palette
        kwargs_dist["y"] = hue
        return kwargs_dist, kwargs_scatter

    def clean_na(self, c1, c2, fillna=None) :

        if fillna is not None :
            self.df[c1] = self.df[c1].fillna(fillna)
            self.df[c2] = self.df[c2].fillna(fillna)

        else :
            self.df = self.df[np.isfinite(self.df[c1])]
            self.df = self.df[np.isfinite(self.df[c2])]

    def init_ui(self, box_ratio, spacers, size) :
        # http://matplotlib.org/api/gridspec_api.html

        gs = gridspec.GridSpec(2, 2, width_ratios=box_ratio[0], height_ratios=box_ratio[1],
        hspace=spacers[0], wspace=spacers[1])

        self.ax_top_dist = self.fig.add_subplot(gs[0])
        self.ax_right_dist = self.fig.add_subplot(gs[3])
        self.ax_scatter = self.fig.add_subplot(gs[2])
        self.set_square(size)

    def draw_scatter(self, c1, c2, xlim=None, ylim=None, ** kwargs) :
        graph = Graph(ax=self.ax_scatter, data=self.df)
        kwargs = {"fit_reg" : False, ** kwargs}
        graph.shs.scatterplot(col1=c1, col2=c2, ** kwargs)
        graph.set_ax_lim(xlim, ylim)

    def draw_dists(self, c1, c2, method, ** kwargs) :
        
        c1v =  self.df[c1]
        c2v =  self.df[c2]

        used_kwargs = {}
        used_kwargs.update(kwargs)

        graph = Graph(ax=self.ax_top_dist, data=self.df)
        getattr(graph.sns, method)(c1v, ** used_kwargs)
        graph.set_labels("", "")
        graph.transform_xticks(lambda x : "")
        graph.remove_legend()
        self.ax_top_dist.set_xlim(self.ax_scatter.get_xlim())

        if "y" in used_kwargs :
            used_kwargs["x"], used_kwargs["y"] = used_kwargs["y"], c2v
        else :
            used_kwargs["orient"] = "v"
            used_kwargs["x"] = c2v

        graph = Graph(ax=self.ax_right_dist, data=self.df)
        getattr(graph.sns, method)(** used_kwargs) 
        graph.set_labels("", "")
        graph.transform_xticks(rotation=90)
        graph.transform_yticks(lambda x : "")
        graph.remove_legend()
        self.ax_right_dist.set_ylim(self.ax_scatter.get_ylim())

    def get_scatter_graph(self) :
        return Graph(self.df, ax=self.ax_scatter)

    def get_top_graph(self) :
        return Graph(self.df, ax=self.ax_top_dist)        

    def get_right_graph(self) :
        return Graph(self.df, ax=self.ax_right_dist)