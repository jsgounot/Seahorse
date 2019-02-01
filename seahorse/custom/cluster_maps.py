# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2018-12-20 13:35:31
# @Last modified by:   jsgounot
# @Last Modified time: 2018-12-21 10:53:11

import pandas as pd
from seahorse import Graph
from seahorse.gwrap import sns

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

def cluster_map(df, rotatey=True, rotatex=True, asgraph=False, ** kwg) :

    clusterobj = sns.clustermap(df, ** kwg)
    graph = Graph(df, ax=clusterobj.ax_heatmap)

    if rotatex : graph.transform_xticks(rotation=90)
    if rotatey : graph.transform_yticks(rotation=0)            

    return graph if asgraph else clusterobj

def genome_map(data, x, y, xhue, value, na=0, use_chroname=True, ytick_rot=0, kwargs_line={}, asgraph=False, * args, ** kwargs) :
    # Change to a class with a Fig !

    df = data
    df = df.sort_values([y, xhue, x])

    subdf = df[[xhue, x]].drop_duplicates([xhue, x]).sort_values([xhue, x])
    additionner = subdf.groupby(xhue)[x].max().cumsum().shift(1).fillna(0).to_dict()
    subdf["gposi"] = subdf.apply(lambda row : additionner[row[xhue]] + row[x], axis=1)
    
    ncols = subdf.groupby(xhue).size().rename("size").reset_index().sort_values(xhue)
    ncols["csize"] = (ncols["size"] - 1).cumsum() + 1
    ncols["tickpos"] = ncols["csize"] - ncols["size"] / 2
    ticksinfo = ncols.set_index(xhue)["tickpos"].to_dict()
    ticksinfo = [(ticksinfo[name], name) for name in sorted(ticksinfo, key = lambda x : ticksinfo[x])]

    df = df.merge(subdf, on=[x, xhue], how="left")
    df = pd.pivot_table(df, index=y, values=value, columns="gposi").fillna(na)
    cluster_grid = sns.clustermap(* args, col_cluster=False, data=df, ** kwargs)

    # Remove labels
    graph = Graph(ax=cluster_grid.ax_heatmap)
    if use_chroname : graph.set_xticks(* list(zip(* ticksinfo)), rotation=45)
    graph.transform_yticks(rotation=ytick_rot)

    # add dashed line
    kwargs_line = {"color" : "black", "linestyle" : "--", ** kwargs_line}
    msizes = ncols.set_index(xhue)["csize"].to_dict()
    for contig, msize in msizes.items() :
        if msize == max(msizes.values()) : continue
        cluster_grid.ax_heatmap.axvline(x=msize, ** kwargs_line)

    res = Graph(data, ax=cluster_grid.ax_heatmap) if asgraph else cluster_grid
    return res