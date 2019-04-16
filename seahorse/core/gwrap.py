import os, copy
import numpy as np
import pandas as pd

from seahorse import sns
from seahorse.core import graphfun
from seahorse.core import constants
from seahorse.core import graph

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

        # doing that we erase binds keys which 
        # are found in kwargs, such as data

        kwargs = {** binds, ** kwargs}
        return getattr(self.lib, funname)(* args, ** kwargs)

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
            obj = getattr(graph.Graph(ax=element), name) if self.graph else getattr(element, name)
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
            except IndexError : raise IndexError("Not enough axes available")
            fun(name, subdf, ax, ** kwargs)
