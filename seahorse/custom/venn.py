# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-02-14 09:46:03
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 16:09:36

try :
    from matplotlib_venn import venn2, venn3, venn2_unweighted, venn3_unweighted
except ImportError :
    print ("Matplotlib venn package not found (optional)")
    print ("Download command line : pip install matplotlib-venn")

from seahorse.core.gwrap import sns

def venn_dic(data, ax, weighted=True, colors=None) :

    if len(data) < 2 :
        print ("You need at least 2 unique values in your hue column to produce a venn diagram")
        return

    if len(data) > 3 :
        print ("You can't produce a venn diagram with more than 3 unique values in your hue column")
        print ("Think about more elaborate option such as PyUpset")
        return

    names, values = [], []
    for name, elements in data.items() :
        names.append(name)
        values.append(set(elements))

    process_venn(ax, names, values, weighted, colors)

def venn_df(column, hue, data, ax, weighted=True, colors=None) :
    
    counting_unique = len(data[hue].unique())    

    if counting_unique < 1 :
        print ("You need at least 2 unique values in your hue column to produce a venn diagram")
        return
    if counting_unique > 3 :
        print ("You can't produce a venn diagram with more than 3 unique values in your hue column")
        print ("Think about more elaborate option such as PyUpset")
        return

    names, values = [], []
    for name, sdf in data.groupby(hue) :
        names.append(name)
        values.append(set(sdf[column]))

    process_venn(ax, names, values, weighted, colors)

def process_venn(ax, names, values, weighted, colors=None) :
    colors = colors or sns.color_palette()

    if len(names) == 2 :
        func = venn2 if weighted else venn2_unweighted
        return func(subsets=values, set_labels=names, ax=ax, set_colors=colors[:2])
    if len(names) == 3 :
        func = venn3 if weighted else venn3_unweighted
        return func(subsets=values, set_labels=names, ax=ax, set_colors=colors[:3])