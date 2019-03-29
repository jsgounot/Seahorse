# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-12-20 13:35:31
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 16:10:10

import pandas as pd
from seahorse import Fig, Graph, cmap_from_color
from seahorse.core.gwrap import sns

class ClusterMap(Fig) :

    def __init__(self, * args, rotatey=0, rotatex=90, rm_yticks=False,
        width_ratios=None, height_ratios=None, ** kwargs) :
        
        self.run_cmap(* args, ** kwargs)

        # we cannot use get here since we get a Value Error
        # if the value is in kwargs
        # ValueError: The truth value of a DataFrame is ambiguous

        try : self.raw_data = kwargs["data"]
        except KeyError : self.raw_data = args[0]

        graph = self.heatmap_graph
        if rm_yticks : graph.remove_yticks()
        if rotatex is not None : graph.transform_xticklabels(rotation=rotatex)
        if rotatey is not None : graph.transform_yticklabels(rotation=rotatey)

        if width_ratios : self.clusterobj.gs.set_width_ratios(width_ratios)
        if height_ratios : self.clusterobj.gs.set_height_ratios(height_ratios)

        self.fig = self.heatmap_ax.get_figure()

    def run_cmap(self, * args, ** kwargs) :
        self.clusterobj = sns.clustermap(* args, ** kwargs)

    @staticmethod
    def drop_same_rows(df) :
        # Maybe there is a better way to do that
        return pd.DataFrame((row for idx, row in df.iterrows() if row.nunique() != 1))

    @property
    def heatmap_graph(self):
        return Graph(ax=self.heatmap_ax)
    
    @property
    def heatmap_ax(self):
        return self.clusterobj.ax_heatmap

    @property
    def cax(self):
        return self.clusterobj.cax
    
    @property
    def data(self):
        return ClusterMap.reorder_df_clustermap(self.raw_data,
            self.clusterobj)
    
    @staticmethod
    def reorder_df_clustermap(df, clusterobj) :
    
        try : 
            indexs = [list(df.index)[idx] for idx in clusterobj.dendrogram_row.reordered_ind]
            df = df.reindex(indexs)

        except AttributeError :
            pass

        try : 
            columns = [list(df.columns)[idx] for idx in clusterobj.dendrogram_col.reordered_ind]
            df = df[columns]

        except AttributeError :
            pass

        return df

class DiscreteClusterMap(ClusterMap) :

    def __init__(self, data, * args, colors=None, ** kwargs) :

        values = sorted(set.union(* [set(data[column]) for column in data.columns]))[::-1]

        if colors is None : 
            colors_flat = sns.cubehelix_palette(len(values))
            cmap = cmap_from_color(colors_flat)
            
        else :
            colors_flat = [colors[value] for value in values]
            cmap = cmap_from_color(colors_flat)

        kwargs["cmap"] = cmap

        ticks = self.cbar_ticks(values)
        cbar_kws = kwargs.setdefault("cbar_kws", {})
        cbar_kws["ticks"] = ticks

        # Now we have to replace values in order to fit in the colorbar space (-1, 1)
        vfun = lambda cell : values.index(cell)
        transformed_data = data.applymap(vfun)

        super().__init__(transformed_data, * args, ** kwargs)
        
        self.raw_data = data
        self.cax.set_yticklabels(values)

    @staticmethod
    def cbar_ticks(values) :
        # return the ticks position for a discret cbar
        # usually the cbar goes to -1 to 1

        cellsize = (len(values) - 1) / len(values)
        subsize = cellsize / 2
        return [(idx * cellsize) - subsize for idx in range(1, len(values) + 1)]

class GenomeMap(ClusterMap) :

    def __init__(self, data, x, y, xhue, value, na_value=0, use_chroname=True, 
        rotatey=True, rotatex=True, kwargs_line={}, * args, ** kwargs) :

        kwargs["data"] = self.transform_data(data, x, y, xhue, value, na_value)
        kwargs["col_cluster"] = False

        kwargs["rotatex"] = rotatex
        kwargs["rotatey"] = rotatey

        super().__init__(* args, ** kwargs)

        if use_chroname : self.heatmap_graph.set_xticks(* list(zip(* self.ticksinfo)), rotation=45)
        self.set_lines(xhue, ** kwargs_line)

    def transform_data(self, data, x, y, xhue, value, na_value) :

        df = data
        df = df.sort_values([y, xhue, x])

        subdf = df[[xhue, x]].drop_duplicates([xhue, x]).sort_values([xhue, x])
        additionner = subdf.groupby(xhue)[x].max().cumsum().shift(1).fillna(0).to_dict()
        subdf["gposi"] = subdf.apply(lambda row : additionner[row[xhue]] + row[x], axis=1)
        
        self.ncols = subdf.groupby(xhue).size().rename("size").reset_index().sort_values(xhue)
        self.ncols["csize"] = (self.ncols["size"] - 1).cumsum() + 1
        self.ncols["tickpos"] = self.ncols["csize"] - self.ncols["size"] / 2
        
        ticksinfo = self.ncols.set_index(xhue)["tickpos"].to_dict()
        self.ticksinfo = [(ticksinfo[name], name) for name in sorted(ticksinfo, key = lambda x : ticksinfo[x])]

        df = df.merge(subdf, on=[x, xhue], how="left")
        return pd.pivot_table(df, index=y, values=value, columns="gposi").fillna(na_value)
        
    def set_lines(self, xhue, ** kwargs) :

        kwargs = {"color" : "black", "linestyle" : "--", ** kwargs}
        msizes = self.ncols.set_index(xhue)["csize"].to_dict()

        for contig, msize in msizes.items() :
            if msize == max(msizes.values()) : continue
            self.heatmap_ax.axvline(x=msize, ** kwargs)
