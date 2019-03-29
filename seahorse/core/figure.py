# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-03-29 15:53:50
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 16:26:31

import os

import pylab as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from seahorse.core import constants

class Fig() :
    # class to manage the mpl figure object
    # such as the figure size, etc

    def __init__(self, fig) :
        self.create_fig()

    def create_fig(self) :

        # Problem here is how pylab handles multiple figures
        # If plt.figure is used, sometimes (for example when using gridspec), you can have weird interactions
        # between different graphs
        # However if you use Figure() it works well but you cannot use the plt.show() function anymore

        if constants.SHOWMODE :
            self.fig = plt.figure()

        else :
            self.fig = Figure()
            self.canvas = constants.FCanva(self.fig)

        if not constants.SHOWMODE or not constants.DEFAULTRES_SHOWMODE :
            self.set_size_inches(* constants.DEFAULTRES)
    
        return self.fig

    def get_size(self, px=False) :
        size = self.fig.get_size_inches()
        if px : size = size * self.fig.dpi
        return size

    def set_vertical(self, ratio=2, vsize=10) :
        hsize = vsize / ratio
        self.fig.set_size_inches(hsize, vsize)

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

        if constants.SAVETAB :
            
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