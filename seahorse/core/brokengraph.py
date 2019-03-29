# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2019-03-29 15:52:47
# @Last modified by:   jsgounot
# @Last Modified time: 2019-03-29 15:54:29

from seahorse.core.figure import Fig

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