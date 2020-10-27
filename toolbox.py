
"""
A collection of indispensible classes and functions for various purposes.
"""

import re
import typing
import string
import enum
import os
import sys
from glob import glob
from pathlib import Path
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


#%% ========================== Plot Helper funcs ========================================

class FigurePolicy(enum.Enum):
    close_create_new = 'Close the previous figure that has the same figname and create a new fresh one'
    add_new = 'Create a new figure with same name but with added suffix'
    same = 'Grab the figure of the same name'


def get_time_stamp(name=None):
    from datetime import datetime
    _ = datetime.now().strftime('%Y-%m-%d-%I-%M-%S-%p-%f')
    if name:
        name = name + '_' + _
    else:
        name = _
    return name


class FigureManager:
    """
    Handles figures of matplotlib.
    """
    def __init__(self, info_loc=None, figure_policy=FigurePolicy.same):
        self.figure_policy = figure_policy
        self.fig = self.ax = self.event = None
        self.cmaps = Cycle(plt.colormaps())
        import matplotlib.colors as mcolors
        self.mcolors = list(mcolors.CSS4_COLORS.keys())
        self.facecolor = Cycle(list(mcolors.CSS4_COLORS.values()))
        self.cmaps.set('viridis')
        self.index = self.pause = self.index_max = None
        self.auto_brightness = False
        self.info_loc = [0.8, 0.01] if info_loc is None else info_loc
        self.pix_vals = False
        self.help_menu = {'_-=+[{]}\\': {'help': "Adjust Vmin Vmax. Shift + key applies change to all axes. \\ "
                                                 "toggles auto-brightness ", 'func': self.adjust_brightness},
                          "/": {'help': 'Show/Hide info text', 'func': self.text_info},
                          "h": {'help': 'Show/Hide help menu', 'func': self.show_help},
                          "tyTY": {'help': 'Change color map', 'func': self.change_cmap},
                          '<>': {'help': 'Change figure face color', 'func': self.change_facecolor},
                          "v": {'help': 'Show/Hide pixel values (Toggle)', 'func': self.show_pix_val},
                          'P': {'help': 'Pause/Proceed (Toggle)', 'func': self.pause_func},
                          'r': {'help': 'Replay', 'func': self.replay},
                          '1': {'help': 'Previous Image', 'func': self.previous},
                          '2': {'help': 'Next Image', 'func': self.next},
                          'S': {'help': 'Save Object', 'func': self.save},
                          'c': {'help': 'Show/Hide cursor', 'func': self.show_cursor},
                          'aA': {'help': 'Show/Hide ticks and their labels', 'func': self.show_ticks},
                          'alt+a': {'help': 'Show/Hide annotations', 'func': self.toggle_annotate}}
        # IMPORTANT: add the 'alt/ctrl+key' versions of key after the key in the dictionary above, not before.
        # Otherwise the naked key version will statisfy the condition `is key in this`? in the parser.
        self.message = ''
        self.message_obj = self.cursor = None
        self.annot_flag = False  # one flag for all axes?
        self.boundaries_flag = True

    @staticmethod
    def grid(ax, factor=5, x_or_y='both', color='gray', alpha1=0.5, alpha2=0.25):
        if type(ax) in {list, List, np.ndarray}:
            for an_ax in ax:
                FigureManager.grid(an_ax, factor=factor, x_or_y=x_or_y, color=color, alpha1=alpha1, alpha2=alpha2)
            return None

        # Turning on major grid for both axes.
        ax.grid(which='major', axis='x', color='gray', linewidth=0.5, alpha=alpha1)
        ax.grid(which='major', axis='y', color='gray', linewidth=0.5, alpha=alpha1)
        if x_or_y in {'both', 'x'}:
            xt = ax.get_xticks()  # major ticks
            steps = (xt[1] - xt[0]) / factor
            ax.xaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='x', color=color, linewidth=0.5, alpha=alpha2)
        if x_or_y in {'both', 'y'}:
            yt = ax.get_yticks()  # major ticks
            steps = (yt[1] - yt[0]) / factor
            ax.yaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='y', color=color, linewidth=0.5, alpha=alpha2)

    def maximize_fig(self):
        _ = self
        plt.get_current_fig_manager().window.state('zoom')

    def toggle_annotate(self, event):
        self.annot_flag = not self.annot_flag
        if event.inaxes:
            if event.inaxes.images:
                event.inaxes.images[0].set_picker(True)
                self.message = f"Annotation flag is toggled to {self.annot_flag}"
        if not self.annot_flag:  # if it is off
            pass  # hide all annotations

    def annotate(self, event, axis=None, data=None):
        self.event = event
        e = event.mouseevent
        if axis is None:
            ax = e.inaxes
        else:
            ax = axis
        if ax:
            if not hasattr(ax, 'annot_obj'):   # first time
                ax.annot_obj = ax.annotate("", xy=(0, 0), xytext=(-30, 30),
                                           textcoords="offset points",
                                           arrowprops=dict(arrowstyle="->", color="w", connectionstyle="arc3"),
                                           va="bottom", ha="left", fontsize=10,
                                           bbox=dict(boxstyle="round", fc="w"),)
            else:
                ax.annot_obj.set_visible(self.annot_flag)

            x, y = int(np.round(e.xdata)), int(np.round(e.ydata))
            if data is None:
                z = e.inaxes.images[0].get_array()[y, x]
            else:
                z = data[y, x]
            ax.annot_obj.set_text(f'x:{x}\ny:{y}\nvalue:{z:.3f}')
            ax.annot_obj.xy = (x, y)
            self.fig.canvas.draw_idle()

    def save(self, event):
        _ = event
        Save.pickle('.', obj=self)

    def replay(self, event):
        _ = event
        self.pause = False
        self.index = 0
        self.message = 'Replaying'
        self.animate()

    def pause_func(self, event):
        _ = event
        self.pause = not self.pause
        self.message = f'Pause flag is set to {self.pause}'
        self.animate()

    def previous(self, event):
        _ = event
        self.index = self.index - 1 if self.index > 0 else self.index_max - 1
        self.message = f'Previous {self.index}'
        self.animate()

    def next(self, event):
        _ = event
        self.index = self.index + 1 if self.index < self.index_max - 1 else 0
        self.message = f'Next {self.index}'
        self.animate()

    def animate(self):
        pass  # a method of the artist child class that is inheriting from this class

    def text_info(self, event):
        _ = event
        self.message = ''

    def show_help(self, event):
        _ = event
        default_plt = {"q ": {'help': "Quit Figure."},
                       "Ll": {'help': "change x/y scale to log and back to linear (toggle)"},
                       "Gg": {'help': "Turn on and off x and y grid respectively."},
                       "s ": {'help': "Save Figure"},
                       "f ": {'help': "Toggle Full screen"},
                       "p ": {'help': "Select / Deselect Pan"}}
        figs = plt.get_figlabels()
        if "Keyboard shortcuts" in figs:
            plt.close("Keyboard shortcuts")  # toggle
        else:
            fig = plt.figure(num="Keyboard shortcuts")
            for i, key in enumerate(self.help_menu.keys()):
                fig.text(0.1, 1 - 0.05 * (i+1), f"{key:30s} {self.help_menu[key]['help']}")
            print(pd.DataFrame([[val['help'], key] for key, val in self.help_menu.items()], columns=['Action', 'Key']))
            print(f"\nDefault plt Keys:\n")
            print(pd.DataFrame([[val['help'], key] for key, val in default_plt.items()], columns=['Action', 'Key']))

    def adjust_brightness(self, event):
        ax = event.inaxes
        if ax is not None and ax.images:
            message = 'None'
            if event.key == '\\':
                self.auto_brightness = not self.auto_brightness
                message = f"Auto-brightness flag is set to {self.auto_brightness}"
                if self.auto_brightness:  # this change is only for the current image.
                    im = self.ax.images[0]
                    im.norm.autoscale(im.get_array())
                    # changes to all ims take place in animate as in ImShow and Nifti methods animate.
            vmin, vmax = ax.images[0].get_clim()
            if event.key in '-_':
                message = 'increase vmin'
                vmin += 1
            elif event.key in '[{':
                message = 'decrease vmin'
                vmin -= 1
            elif event.key in '=+':
                message = 'increase vmax'
                vmax += 1
            elif event.key in ']}':
                message = 'decrease vmax'
                vmax -= 1
            self.message = message + '  ' + str(round(vmin, 1)) + '  ' + str(round(vmax, 1))
            if event.key in '_+}{':
                for ax in self.fig.axes:
                    if ax.images:
                        ax.images[0].set_clim((vmin, vmax))
            else:
                if ax.images:
                    ax.images[0].set_clim((vmin, vmax))

    def change_cmap(self, event):
        ax = event.inaxes
        if ax is not None:
            cmap = self.cmaps.next() if event.key in 'tT' else self.cmaps.previous()
            if event.key in 'TY':
                for ax in self.fig.axes:
                    for im in ax.images:
                        im.set_cmap(cmap)
            else:
                for im in ax.images:
                    im.set_cmap(cmap)
            self.message = f"Color map changed to {ax.images[0].cmap.name}"

    def change_facecolor(self, event):
        color = self.facecolor.next() if event.key == '>' else self.facecolor.previous()
        self.fig.set_facecolor(color)
        self.message = f"Figure facecolor was set to {self.mcolors[self.facecolor.get_index()]}"

    def show_pix_val(self, event):
        ax = event.inaxes
        if ax is not None:
            self.pix_vals = not self.pix_vals  # toggle
            self.message = f"Pixel values flag set to {self.pix_vals}"
            if self.pix_vals:
                self.show_pixels_values(ax)
            else:
                while len(ax.texts) > 0:
                    for text in ax.texts:
                        text.remove()

    def process_key(self, event):
        self.event = event  # useful for debugging.
        for key in self.help_menu.keys():
            if event.key in key:
                self.help_menu[key]['func'](event)
                break
        self.update_info_text(self.message)
        if event.key != 'q':  # for smooth quit without throwing errors
            fig = event.canvas.figure  # don't update if you want to quit.
            fig.canvas.draw()

    def update_info_text(self, message):
        if self.message_obj:
            self.message_obj.remove()
        self.message_obj = self.fig.text(*self.info_loc, message, fontsize=8)

    @staticmethod
    def get_nrows_ncols(num_plots, nrows=None, ncols=None):
        if not nrows and not ncols:
            nrows = int(np.floor(np.sqrt(num_plots)))
            ncols = int(np.ceil(np.sqrt(num_plots)))
            while nrows * ncols < num_plots:
                ncols += 1
        elif not ncols and nrows:
            ncols = int(np.ceil(num_plots / nrows))
        elif not nrows and ncols:
            nrows = int(np.ceil(num_plots / ncols))
        else:
            pass
        return nrows, ncols

    def show_cursor(self, event):
        ax = event.inaxes
        if ax:  # don't do this if c was pressed outside an axis.
            if hasattr(ax, 'cursor_'):  # is this the first time?
                if ax.cursor_ is None:
                    from matplotlib import widgets
                    ax.cursor_ = widgets.Cursor(ax=ax, vertOn=True, horizOn=True, color='red', lw=1.0)
                else:  # toggle the cursor.
                    ax.cursor_ = None
                self.message = f'Cursor flag set to {bool(ax.cursor_)}'
            else:  # first call
                ax.cursor_ = None
                self.show_cursor(event)

    def show_ticks(self, event):
        self.boundaries_flag = not self.boundaries_flag
        axis = event.inaxes
        if event.key == 'a':
            if axis:
                # event.inaxes.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(axis)
                self.message = f"Boundaries flag set to {self.boundaries_flag} in {axis}"

        else:
            for ax in self.ax.figure.axes:
                # ax.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(ax)

    @staticmethod
    def toggle_ticks(an_ax, state=None):
        for line in an_ax.get_yticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklabels():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_yticklabels():
            line.set_visible(not line.get_visible() if state is None else state)

    def clear_axes(self):
        for ax in self.ax:
            ax.cla()

    @staticmethod
    def show_pixels_values(ax):
        im = ax.images[0].get_array()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if ymin > ymax:  # default imshow settings
            ymin, ymax = ymax, ymin
        for (j, i), label in np.ndenumerate(im):
            if (xmin < i < xmax) and (ymin < j < ymax):
                ax.text(i, j, np.round(label).__int__(), ha='center', va='center', size=8)

    @staticmethod
    def update(fig_name, obj_name, data=None):
        """Fastest update ever. But, you need access to label name.
        Using this function external to the plotter. But inside the plotter you need to define labels to objects
        The other alternative is to do the update inside the plotter, but it will become very verbose.

        :param fig_name:
        :param obj_name:
        :param data:
        :return:
        """
        obj = FigureManager.findobj(fig_name, obj_name)
        if data is not None:
            obj.set_data(data)
            # update scale:
            # obj.axes.relim()
            # obj.axes.autoscale()

    @staticmethod
    def findobj(fig_name, obj_name):
        if type(fig_name) is str:
            fig = plt.figure(num=fig_name)
        else:
            fig = fig_name
        search_results = fig.findobj(lambda x: x.get_label() == obj_name)
        if len(search_results) > 0:  # list of length 1, 2 ...
            search_results = search_results[0]  # the first one is good enough.
        return search_results

    def get_fig(self, figname='', suffix=None, **kwargs):
        return FigureManager.get_fig_static(self.figure_policy, figname, suffix, **kwargs)

    @staticmethod
    def get_fig_static(figure_policy, figname='', suffix=None, **kwargs):
        """
        :param figure_policy:
        :param figname:
        :param suffix: only relevant if figure_policy is add_new
        :param kwargs:
        :return:
        """
        fig = None
        exist = True if figname in plt.get_figlabels() else False
        if figure_policy is FigurePolicy.same:
            fig = plt.figure(num=figname, **kwargs)
        elif figure_policy is FigurePolicy.add_new:
            if exist:
                new_name = get_time_stamp(figname) if suffix is None else figname + suffix
            else:
                new_name = figname
            fig = plt.figure(num=new_name, **kwargs)
        elif figure_policy is FigurePolicy.close_create_new:
            if exist:
                plt.close(figname)
            fig = plt.figure(num=figname, **kwargs)
        return fig

    def transperent_fig(self):
        self.fig.canvas.manager.window.attributes("-transparentcolor", "white")

    @staticmethod
    def set_ax_size(ax, w, h):
        """ w, h: width, height in inches """
        left = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - left)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    @staticmethod
    def get_ax_size(ax):  # returns axis size in inches.
        w, h = ax.figure.get_size_inches()
        width = ax.figure.subplotpars.right - ax.figure.subplotpars.left
        height = ax.figure.subplotpars.top - ax.figure.subplotpars.bottom
        # width, height = ax.figbox.extents[2:] - ax.figbox.extents[:2]
        return w * width, h * height

    @staticmethod
    def set_ax_to_real_life_size(ax, inch_per_unit=1/25.4):
        limit_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        limit_y = ax.get_ylim()[1] - ax.get_ylim()[0]
        FigureManager.set_ax_size(ax, limit_x * inch_per_unit, limit_y * inch_per_unit)

    @staticmethod
    def try_figure_size():
        fig, ax = plt.subplots()
        x = np.arange(0, 100, 0.01)
        y = np.sin(x) * 100
        ax.plot(x, y)
        ax.axis("square")
        ax.set_xlim(0, 100)
        ax.set_ylim(-100, 100)
        FigureManager.set_ax_to_real_life_size(ax)
        fig.savefig(P.tmp() / "trial.png", dpi=250)

    @staticmethod
    def write(txt, name="text", size=8, **kwargs):
        fig = plt.figure(figsize=(11.69, 8.27), num=name)
        fig.clf()
        fig.text(0.5, 0.5, txt, transform=fig.transFigure, size=size, ha="center", **kwargs)
        return fig

    @staticmethod
    def activate_latex(size=20):
        """Setting up matplotlib"""
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('axes', labelsize=size)
        plt.rc('legend', fontsize=size-5)
        # rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


class SaveType:
    """
    Programming philosophy: this class only worries about saving, and saving only. In other words, the figure must
    be fully prepared beforehand. Names here are only used for the purpose of saving, never putting titles on figures.
    """

    class GenericSave:
        """ You can either pass the figures to be tracked or, pass them dynamically at add method, or,
        add method will capture every figure and axis

        """
        stream = ['clear', 'accumulate', 'update'][0]

        def __init__(self, save_dir=None, save_name=None, watch_figs=None, max_calls=2000, delay=100, **kwargs):
            self.delay = delay
            self.watch_figs = watch_figs
            if watch_figs:
                assert type(watch_figs) is list, "This should be a list"
                if type(watch_figs[0]) is str:
                    self.watch_figs = [plt.figure(num=afig) for afig in watch_figs]

            if save_dir is None:
                save_dir = Path('./').absolute().__str__()
            self.save_name = get_time_stamp(save_name)
            self.save_dir = save_dir
            self.kwargs = kwargs
            self.counter = 0
            self.max = max_calls

        def add(self, fig_names=None, names=None, **kwargs):
            print(f"Saver added frame number {self.counter}", end='\r')
            self.counter += 1
            plt.pause(self.delay * 0.001)
            if self.counter > self.max:
                print('Turning off IO')
                plt.ioff()

            if fig_names:  # name sent explicitly
                self.watch_figs = [plt.figure(fig_name) for fig_name in fig_names]
            else:  # tow choices:
                if self.watch_figs is None:  # None exist ==> add all
                    figure_names = plt.get_figlabels()  # add all.
                    self.watch_figs = [plt.figure(k) for k in figure_names]
                else:  # they exist already.
                    pass

            if names is None:  # individual save name, useful for PNG.
                names = [get_time_stamp(a_figure.get_label()) for a_figure in self.watch_figs]

            for afig, aname in zip(self.watch_figs, names):
                self._save(afig, aname, **kwargs)

        def _save(self, *args, **kwargs):
            pass

    class Null(GenericSave):
        """ Use this when you do not want to save anything. This class will help plot to work faster
        by removing lines of previous plot, so you get live animation cheaply.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fname = self.save_dir

        def finish(self):
            print(f"Nothing saved by {self}")
            return self.fname

    class PDF(GenericSave):
        """For pdf, you just need any figure manager, [update, clear, accumalate], preferabbly fastest.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from matplotlib.backends.backend_pdf import PdfPages
            self.fname = os.path.join(self.save_dir, self.save_name + '.pdf')
            self.pp = PdfPages(self.fname)

        def _save(self, a_fig, a_name, bbox_inches='tight', pad_inches=0.3, **kwargs):
            self.pp.savefig(a_fig, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

        def finish(self, open_result=True):
            print(f"Saving results ...")
            self.pp.close()
            print(f"PDF Saved @", Path(self.fname).absolute().as_uri())
            if open_result:
                import webbrowser as wb
                chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
                wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))
                wb.get('chrome').open(self.fname)
            return self.fname

    class PNG(GenericSave):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_dir = os.path.join(self.save_dir, self.save_name)
            os.makedirs(self.save_dir, exist_ok=True)
            self.fname = self.save_dir

        def _save(self, afigure, aname, dpi=150, **kwargs):
            aname = P(aname).make_python_name()
            afigure.savefig(os.path.join(self.save_dir, aname), bbox_inches='tight', pad_inches=0.3,
                            dpi=dpi, **kwargs)

        def finish(self):
            print(f"PNGs Saved @", Path(self.fname).absolute().as_uri())
            return self.fname

    class GIF(GenericSave):
        """Requirements: same axis must persist, only new objects are drawn inside it.
        This is not harsh as no one wants to add multiple axes on top of each other.
        Next, the objects drawn must not be removed, or updated, instead they should pile up in axis.

        # do not pass names in the add method. names will be extracted from figures.
        # usually it is smoother when adding animate=True to plot or imshow commands for GIF purpose

        Works for images only. Add more .imshow to the same axis, and that's it. imshow will conver up previous images.
        For lines, it will superimpose it and will look ugly.

        If you clear the axis, nothing will be saved. This should not happend.
        The class will automatically detect new lines by their "neo" labels.
        and add them then hide them for the next round.
        Limitation of ArtistAnimation: works on lines and images list attached to figure axes.
        Doesn't work on axes, unless you add large number of them. As such, titles are not incorporated etc.
        """

        def __init__(self, interval=100, **kwargs):
            super().__init__(**kwargs)
            from collections import defaultdict
            self.container = defaultdict(lambda: [])
            self.interval = interval
            self.fname = None  # determined at finish time.

        def _save(self, afigure, aname, cla=False, **kwargs):
            fig_list = self.container[afigure.get_label()]
            subcontainer = []

            search = FigureManager.findobj(afigure, 'neo')
            for item in search:
                item.set_label('processed')
                item.set_visible(False)
                subcontainer += [item]

            fig_list.append(subcontainer)
            # if you want the method coupled with cla being used in main, then it add_line is required for axes.

        def finish(self):
            print("Saving the GIF ....")
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
            for idx, a_fig in enumerate(self.watch_figs):
                ims = self.container[a_fig.get_label()]
                if ims:
                    ani = animation.ArtistAnimation(a_fig, ims,
                                                    interval=self.interval, blit=True, repeat_delay=1000)
                    self.fname = os.path.join(self.save_dir, f'{a_fig.get_label()}_{self.save_name}.gif')
                    ani.save(self.fname, writer=PillowWriter(fps=4))
                    # if you don't specify the writer, it goes to ffmpeg by default then try others if that is not
                    # available, resulting in behaviours that is not consistent across machines.
                    print(f"GIF Saved @", Path(self.fname).absolute().as_uri())
                else:
                    print(f"Nothing to be saved by GIF writer.")
                    return self.fname

    class GIFFileBased(GenericSave):
        def __init__(self, fps=4, dpi=100, bitrate=1800, _type='GIFFileBased', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBased':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBased':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBased':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'
            self.writer = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)
            assert self.watch_figs, "No figure was sent during instantiation of saver, therefore the writer cannot" \
                                    "be setup. Did you mean to use an autosaver?"
            self.writer.setup(fig=self.watch_figs[0], outfile=self.fname, dpi=dpi)

        def _save(self, afig, aname, **kwargs):
            self.writer.grab_frame(**kwargs)

        def finish(self):
            print('Saving results ...')
            self.writer.finish()
            print(f"Saved @", Path(self.fname).absolute().as_uri())
            return self.fname

    class GIFPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    """
    Parses the data automatically.
    For all subclasses, you need to provide a plotter class with animate method implemetend.
    You also need to have .fig attribute.
    """

    class GenericAuto(GenericSave):
        save_type = 'auto'

        def __init__(self, plotter_class, data, names_list=None, **kwargs):
            super().__init__(**kwargs)
            self.plotter_class = plotter_class
            self.data = data
            self.names_list = names_list
            self.kwargs = kwargs
            self.data_gen = None
            self.saver = None
            self.plotter = None

        def animate(self):
            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.data_gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **self.kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them
            for idx, datum in tqdm(enumerate(self.data_gen())):
                self.plotter.animate(datum)
                self.saver.add(names=[self.names_list[idx]])
            self.saver.finish()

    class GIFAuto(GenericAuto):
        def __init__(self, plotter_class, data, interval=500, extension='gif', fps=4, **kwargs):
            super().__init__(plotter_class, data, **kwargs)
            writer = None
            from matplotlib import animation
            if extension == 'gif':
                writer = animation.PillowWriter(fps=fps)
            elif extension == 'mp4':
                writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=2500)

            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **kwargs)
            plt.pause(self.delay * 0.001)  # give time for figures to show up before updating them
            self.ani = animation.FuncAnimation(self.plotter.fig, self.plotter.animate, frames=self.gen,
                                               interval=interval, repeat_delay=1500, fargs=None,
                                               cache_frame_data=True, save_count=10000)
            fname = f"{os.path.join(self.save_dir, self.save_name)}.{extension}"
            self.fname = fname
            self.ani.save(filename=fname, writer=writer)
            print(f"Saved @", Path(self.fname).absolute().as_uri())

    class PDFAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PDF(**kwargs)
            self.animate()

    class PNGAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PNG(**kwargs)
            self.save_dir = self.saver.save_dir
            self.animate()
            self.fname = self.saver.fname

    class NullAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.Null(**kwargs)
            self.fname = self.saver.fname
            self.animate()

    class GIFFileBasedAuto(GenericAuto):
        def __init__(self, plotter_class, data, fps=4, dpi=150, bitrate=2500,
                     _type='GIFFileBasedAuto', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBasedAuto':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBasedAuto':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBasedAuto':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'

            self.saver = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)

            def gen_function():
                for i in zip(*data):
                    yield i

            self.data = gen_function
            self.plotter = plotter_class(*[piece[0] for piece in data], **kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them
            with self.saver.saving(fig=self.plotter.fig, outfile=self.fname, dpi=dpi):
                for datum in tqdm(self.data()):
                    self.plotter.animate(datum)
                    self.saver.grab_frame()
                    plt.pause(self.delay * 0.001)
            print(f"Results saved successfully @ {self.fname}")

    class GIFPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)


class VisibilityViewer(FigureManager):
    artist = ['internal', 'external'][1]
    parser = ['internal', 'external'][1]
    stream = ['clear', 'accumulate', 'update'][1]
    """
    **Viewer Building Philosophy**: 
    
    Viewer should act as Saver and Browser:

    * How is the data viewed:
    
        * Can either be an artist himself, as in ImShow.
        * external artist is required to view the data (especially non image data)

    * Data parsing:

        * internal for loop to go through all the dataset passed.
            # Allows manual control over parsing.
        * external for loop. It should have add method.
            # Manual control only takes place after the external loop is over. #TODO parallelize this.

    * Refresh mechanism.

        * Clear the axis.
        * accumulate, using visibility to hide previous axes.
        * The artist has an update method.

    The artist has to have:
    
        * fig, ax, txt attributes. ax and txt should be lists.
        * the ax and txt attributes should always belong to the same figure.
    
    Here and in all Visibility classes, the artist is assumed to be always creating new axes along the way.
    """
    def __init__(self, artist=None, hide_artist_axes=True):
        """
        This class works on hiding axes shown on a plot, so that a new plot can be drawn.
        Hiding is done via the method `add`.
        Thus, an external loop is required to parse through the plots one by one.
        Once the entire loop is finished, you can browse through the plots with the keyboard
        Animation is done bia method `animate`

        :param artist: A class that draws on one figure. It should have `.fig` attribute.
                       Can either be passed during instantiation, or everytime when `add` is called.
        :param hide_artist_axes:
        """
        super().__init__()
        self.index = -1
        self.index_max = 0
        self.current = None
        self.axes_repo = []
        self.texts_repo = []
        self.fig = None
        if artist:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)
            self.add(artist=artist, hide_artist_axes=hide_artist_axes)

    def add(self, artist=None, increment_index=True, hide_artist_axes=True):
        if artist is not None:
            self.artist = artist
        if self.fig is None:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)

        if increment_index:
            self.index += 1
            self.index_max += 1
        self.current = self.index
        self.axes_repo.append(self.artist.ax if type(self.artist.ax) is list else self.artist.ax.tolist())
        self.texts_repo.append(self.artist.txt if type(self.artist.txt) is list else self.artist.txt.tolist())
        print(f"VViewer added plot number {self.index}", end='\r')
        if hide_artist_axes:
            self.hide_artist_axes()

    def hide_artist_axes(self):
        for ax in self.artist.ax:
            ax.set_visible(False)
        for text in self.artist.txt:
            text.set_visible(False)

    def finish(self):  # simply: undo the last hiding
        self.current = self.index
        self.animate()

    def animate(self):
        # remove current axes and set self.index as visible.
        for ax in self.axes_repo[self.current]:
            ax.set_visible(False)
        for text in self.texts_repo[self.current]:
            text.set_visible(False)
        for ax in self.axes_repo[self.index]:
            ax.set_visible(True)
        for text in self.texts_repo[self.index]:
            text.set_visible(True)
        self.current = self.index
        self.fig.canvas.draw()


class VisibilityViewerAuto(VisibilityViewer):
    def __init__(self, data=None, artist=None, memorize=False, transpose=True, save_type=SaveType.Null,
                 save_dir=None, save_name=None, delay=1,
                 titles=None, legends=None, x_labels=None, pause=True, **kwargs):
        """
        The difference between this class and `VisibilityViewer` is that here the parsing of data is done
        internally, hence the suffix `Auto`.

        :param data: shoud be of the form [[ip1 list], [ip2 list], ...]
                     i.e. NumInputs x LengthInputs x Input
        :param artist: an instance of a class that subclasses `Artist`
        :param memorize: if set to True, then axes are hidden and shown again, otherwise, plots constructed freshly
                         every time they're shown (axes are cleaned instead of hidden)
        """
        self.kwargs = kwargs
        self.memorize = memorize
        self.max_index_memorized = 0

        if transpose:
            data = np.array(list(zip(*data)))
        self.data = data
        self.legends = legends if legends is not None else [f"Curve {i}" for i in range(len(self.data))]
        self.titles = titles if titles is not None else np.arange(len(self.data))
        self.lables = x_labels

        if artist is None:
            artist = Artist(*self.data[0], title=self.titles[0], legends=self.legends[0], create_new_axes=True,
                            **kwargs)
        else:
            artist.plot(*self.data[0], title=self.titles[0], legends=self.legends[0])
            if memorize:
                assert artist.create_new_axes is True, "Auto Viewer is based on hiding and showing and requires new " \
                                                       "axes from the artist with every plot"
        self.artist = artist
        super().__init__(artist=self.artist, hide_artist_axes=False)
        self.index_max = len(self.data)
        self.pause = pause
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay)
        self.fname = None

    def animate(self):
        for i in range(self.index, self.index_max):
            datum = self.data[i]
            if self.memorize:  # ==> plot and use .add() method
                if self.index > self.max_index_memorized:  # a new plot never done before
                    self.hide_artist_axes()
                    self.artist.plot(*datum, title=self.titles[i], legends=self.legends[i])
                    self.add(increment_index=False, hide_artist_axes=False)  # index incremented via press_key manually
                    self.max_index_memorized += 1
                else:  # already seen this plot before ==> use animate method of parent class to hide and show,
                    # not plot and add
                    # print(f"current = {self.current}")
                    super().animate()
            else:
                self.fig.clf()  # instead of making previous axis invisible, delete it completely.
                self.artist.plot(*datum, title=self.titles[i], legends=self.legends[i])
                # replot the new data point on a new axis.
            self.saver.add()
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    @staticmethod
    def test():
        return VisibilityViewerAuto(data=np.random.randn(1, 10, 100, 3))


class ImShow(FigureManager):
    artist = ['internal', 'external'][0]
    parser = ['internal', 'external'][0]
    stream = ['clear', 'accumulate', 'update'][2]

    def __init__(self, *images_list: typing.Union[list, np.ndarray], sup_titles=None, sub_labels=None, labels=None,
                 save_type=SaveType.Null, save_name=None, save_dir=None, save_kwargs=None,
                 subplots_adjust=None, gridspec=None, tight=True, info_loc=None,
                 nrows=None, ncols=None, ax=None,
                 figsize=None, figname='im_show', figure_policy=FigurePolicy.add_new,
                 auto_brightness=True, delay=200, pause=False,
                 **kwargs):
        """
        :param images_list: arbitrary number of image lists separated by comma, say N.
        :param sup_titles: Titles for frames. Must have a length equal to number of images in each list, say M.
        :param sub_labels: Must have a length = M, and in each entry there should be N labels.
        :param labels: if labels are sent via this keyword, they will be repeated for all freames.
        :param save_type:
        :param save_dir:
        :param save_name:
        :param nrows:
        :param ncols:
        :param delay:
        :param kwargs: passed to imshow

        Design point: The expected inputs are made to be consistent with image_list passed. Labels and titles passed
        should have similar structure. The function internally process them.

        Tip: Use np.arrray_split to get sublists and have multiple plots per frame. Useful for very long lists.
        """
        super(ImShow, self).__init__(info_loc=info_loc)

        num_plots = len(images_list)  # Number of images in each plot
        self.num_plots = num_plots
        lengths = [len(images_list[i]) for i in range(num_plots)]
        self.index_max = min(lengths)
        nrows, ncols = self.get_nrows_ncols(num_plots, nrows, ncols)

        # Pad zero images for lists that have differnt length from the max.
        # images_list = list(images_list)  # now it is mutable, unlike tuple.
        # for i, a_list in enumerate(images_list):
        #     diff = self.num_images - len(a_list)
        #     if diff > 0:
        #         for _ in range(diff):
        #             if type(a_list) is list:
        #                 a_list.append(np.zeros_like(a_list[0]))
        #             else:  # numpy array
        #                 a_list = np.concatenate([a_list, [a_list[0]]])
        #     images_list[i] = a_list
        # # As far as labels are concerned, None is typed, if length is passed.

        # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        # axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        # bnext = Button(axnext, 'Next')
        # bnext.on_clicked(callback.next)
        # bprev = Button(axprev, 'Previous')
        # bprev.on_clicked(callback.prev)

        if sup_titles is None:
            sup_titles = [str(i) for i in np.arange(self.index_max)]

        if labels:
            sub_labels = [[a_label for _ in np.arange(self.index_max)] for a_label in labels]
        elif sub_labels is None:
            sub_labels = [[str(i) for i in np.arange(self.index_max)] for _ in range(self.num_plots)]

        self.image_list = images_list
        self.sub_labels = sub_labels
        self.titles = sup_titles
        self.delay = delay
        self.fname = None
        self.kwargs = kwargs
        self.event = None
        self.cmaps = Cycle(plt.colormaps())
        self.cmaps.set('viridis')
        self.auto_brightness = auto_brightness
        if ax is None:
            self.figure_policy = figure_policy
            self.fig = self.get_fig(figname=figname,
                                    figsize=(14, 9) if figsize is None else figsize, facecolor='white')
            if figsize is None:
                plt.get_current_fig_manager().full_screen_toggle()
                # .window.showMaximized()  # state('zoom')
                # plt.get_current_fig_manager().window.setGeometry(800,70,1000,900)
            if gridspec is not None:
                gs = self.fig.add_gridspec(gridspec[0])
                self.ax = []
                for ags in gs[1:]:
                    self.ax.append(self.fig.add_subplot(gs[ags[0], ags[1]]))
            else:
                self.ax = self.fig.subplots(nrows=nrows, ncols=ncols)
        else:
            self.ax = ax
            try:
                self.fig = ax[0].figure
            except TypeError:  # Not subscriptable, single axis.
                self.fig = ax.figure

        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect("pick_event", self.annotate)
        if tight:
            self.fig.tight_layout()
        if subplots_adjust is not None:
            self.fig.subplots_adjust(**subplots_adjust)

        # if save_type.parser == "internal":
        #     raise TypeError("Requires external data parser")
        if save_kwargs is None:
            save_kwargs = {}
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay, **save_kwargs)
        if nrows == 1 and ncols == 1:
            self.ax = [self.ax]  # make a list out of it.
        else:
            self.ax = self.ax.ravel()  # make a 1D  list out of a 2D array.
        for an_ax in self.ax:
            # an_ax.set_xticks([])
            # an_ax.set_yticks([])
            self.toggle_ticks(an_ax, state=False)
        self.transposed_images = [images for images in zip(*images_list)]
        self.transposed_sublabels = [labels for labels in zip(*sub_labels)]

        self.ims = []  # container for images.
        self.pause = pause
        self.index = 0
        self.animate()

    def animate(self):
        for i in range(self.index, self.index_max):
            for j, (an_image, a_label, an_ax) in enumerate(zip(self.transposed_images[i],
                                                               self.transposed_sublabels[i],
                                                               self.ax)):
                # with zipping, the shortest of the three, will stop the loop.
                if i == 0 and self.ims.__len__() < self.num_plots:
                    im = an_ax.imshow(an_image, animated=True, **self.kwargs)
                    self.ims.append(im)
                else:
                    self.ims[j].set_data(an_image)
                if self.auto_brightness:
                    self.ims[j].norm.autoscale(an_image)
                an_ax.set_xlabel(f'{a_label}')
            self.fig.suptitle(self.titles[i], fontsize=8)
            self.saver.add(names=[self.titles[i]])
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    def annotate(self, event, axis=None, data=None):
        for ax in self.ax:
            super().annotate(event, axis=ax, data=ax.images[0].get_array())

    @classmethod
    def from_saved_images_path_lists(cls, *image_list, **kwargs):
        images = []
        sub_labels = []
        for alist in image_list:
            image_subcontainer = []
            label_subcontainer = []
            for an_image in alist:
                image_subcontainer.append(plt.imread(an_image))
                label_subcontainer.append(os.path.basename(an_image).split('.')[0])
            images.append(image_subcontainer)
            sub_labels.append(label_subcontainer)
        return cls(*images, sub_labels=sub_labels, **kwargs)

    @classmethod
    def from_directories(cls, *directories, extension='png', **kwargs):
        paths = []
        for a_dir in directories:
            paths.append(P(a_dir).myglob(f"*.{extension}", win_order=True))
        return cls.from_saved_images_path_lists(*paths, **kwargs)

    @classmethod
    def from_saved(cls, *things, **kwargs):
        exmaple_item = things[0]
        if isinstance(exmaple_item, list):
            return cls.from_saved_images_path_lists(*things, **kwargs)
        else:
            return cls.from_directories(*things, **kwargs)

    @staticmethod
    def cm(im, nrows=3, ncols=7, **kwargs):
        """ Useful for looking at one image in multiple cmaps

        :param im:
        :param nrows:
        :param ncols:
        :param kwargs:
        :return:
        """
        styles = plt.colormaps()
        colored = [plt.get_cmap(style)(im) for style in styles]
        splitted = np.array_split(colored, nrows * ncols)
        labels = np.array_split(styles, nrows * ncols)
        _ = ImShow(*splitted, nrows=nrows, ncols=ncols, sub_labels=labels, **kwargs)
        return colored

    @staticmethod
    def test():
        return ImShow(*np.random.randn(12, 101, 100, 100))

    @classmethod
    def complex(cls, data, pause=True, **kwargs):
        return cls(data.real, data.imag, np.angle(data), abs(data), labels=['Real Part', 'Imaginary Part',
                                                                            'Angle in Radians', 'Absolute Value'],
                   pause=pause, **kwargs)

    @staticmethod
    def resize(path, m, n):
        from skimage.transform import resize
        image = plt.imread(path)
        image_resized = resize(image, (m, n), anti_aliasing=True)
        plt.imsave(path, image_resized)


def plot_dict(adict):
    obj = Artist(figname='Dictionary Plot')
    for key, val in zip(adict.keys(), adict.values()):
        obj.plot(val, label=key)
    obj.fig.legend()
    return obj


class Artist(FigureManager):
    def __init__(self, *args, ax=None, figname='Graph', title='', label='curve', style='seaborn',
                 create_new_axes=False, figure_policy=FigurePolicy.add_new, figsize=(7, 4), **kwargs):
        super().__init__(figure_policy=figure_policy)
        self.style = style
        # self.kwargs = kwargs
        self.title = title
        self.args = args
        self.line = self.cursor = self.check_b = None
        if ax is None:  # create a figure
            with plt.style.context(style=self.style):
                self.fig = self.get_fig(figname, figsize=figsize)
        else:  # use the passed axis
            self.ax = ax
            self.fig = ax[0].figure

        if len(args):  # if there's something to plot in the init
            if not ax:  # no ax sent but we need to plot, we need an ax, plot will soon call get_axes.
                self.create_new_axes = True  # just for the first time in this init method.
            self.plot(*self.args, label=label, title=title, **kwargs)
        else:  # nothing to be plotted in the init
            if not create_new_axes:  # are we going to ever create new axes?
                self.create_new_axes = True  # if not then let's create one now.
                self.get_axes()
        self.create_new_axes = create_new_axes
        self.visibility_ax = [0.01, 0.05, 0.2, 0.15]
        self.txt = []

    def plot(self, *args, **kwargs):
        self.get_axes()
        self.accessorize(*args, **kwargs)

    def accessorize(self, *args, legends=None, title=None, **kwargs):
        self.line = self.ax[0].plot(*args, **kwargs)
        if legends is not None:
            self.ax[0].legend(legends)
        if title is not None:
            self.ax[0].set_title(title)
        self.ax[0].grid('on')

    def get_axes(self):
        if self.create_new_axes:
            axis = self.fig.subplots()
            self.ax = np.array([axis])
        else:  # use same old axes
            pass

    def suptitle(self, title):
        self.txt = [self.fig.text(0.5, 0.98, title, ha='center', va='center', size=9)]

    def visibility(self):
        from matplotlib.widgets import CheckButtons
        self.fig.subplots_adjust(left=0.3)
        self.visibility_ax[-1] = 0.05 * len(self.ax.lines)
        rax = self.fig.add_axes(self.visibility_ax)
        labels = [str(line.get_label()) for line in self.ax.lines]
        visibility = [line.get_visible() for line in self.ax.lines]
        self.check_b = CheckButtons(rax, labels, visibility)

        def func(label):
            index = labels.index(label)
            self.ax.lines[index].set_visible(not self.ax.lines[index].get_visible())
            self.fig.canvas.draw()
        self.check_b.on_clicked(func)

    @staticmethod
    def styler(plot_gen):
        styles = plt.style.available
        for astyle in styles:
            with plt.style.context(style=astyle):
                plot_gen()
                plt.title(astyle)
                plt.pause(1)
                plt.cla()


class Cycle:
    def __init__(self, c):
        self._c = c
        self._index = -1

    def next(self):
        self._index += 1
        if self._index >= len(self._c):
            self._index = 0
        return self._c[self._index]

    def previous(self):
        self._index -= 1
        if self._index < 0:
            self._index = len(self._c)-1
        return self._c[self._index]

    def set(self, value):
        self._index = self._c.index(value)

    def get(self):
        return self._c[self._index]

    def get_index(self):
        return self._index

    def set_index(self, index):
        self._index = index


#%% ========================== File Management  =========================================


class Browse(object):
    def __init__(self, path, directory=True):
        # Create an attribute in __dict__ for each child
        self.__path__ = path
        if directory:
            sub_paths = glob(os.path.join(path, '*'))
            names = [os.path.basename(i) for i in sub_paths]
            # this is better than listdir, gives consistent results with glob
            for file, full in zip(names, sub_paths):
                key = P(file).make_python_name()
                setattr(self, 'FDR_' + key if os.path.isdir(full) else 'FLE_' + key,
                        full if os.path.isdir(full) else Browse(full, False))

    def __getattribute__(self, name):
        if name == '__path__':
            return super().__getattribute__(name)
        d = super().__getattribute__('__dict__')
        if name in d:
            child = d[name]
            if isinstance(child, str):
                child = Browse(child)
                setattr(self, name, child)
            return child
        return super().__getattribute__(name)

    def __repr__(self):
        return self.__path__

    def __str__(self):
        return self.__path__


class P(type(Path()), Path):
    """Path Class: Designed with one goal in mind: any operation on paths MUST NOT take more than one line of code.
    """

    def size(self, units='mb'):
        factor = {'b': 1, 'kb': 1024, 'mb': 1024**2, 'gb': 1024**3}[units]
        if self.is_file():
            total_size = self.stat().st_size
        elif self.is_dir():
            results = self.rglob("*")
            total_size = 0
            for item in results:
                if item.is_file():
                    total_size += item.stat().st_size
        else:
            raise TypeError("This thing is not a file nor a folder.")
        return round(total_size / factor, 1)

    def get_num(self, astring=None):
        if astring is None:
            astring = self.stem
        return int("".join(filter(str.isdigit, str(astring))))

    def make_python_name(self, astring=None):
        if astring is None:
            astring = self.stem
        return re.sub(r'^(?=\d)|\W', '_', str(astring))

    @property
    def trunk(self):
        """ useful if you have multiple dots in file name where .stem fails.
        """
        return self.name.split('.')[0]

    def __add__(self, name):
        return self.parent.joinpath(self.stem + name)

    def prepend(self, name, stem=False):
        """Add extra text before file name
        e.g: blah\blah.extenion ==> becomes ==> blah/name_blah.extension
        """
        if stem:
            return self.parent.joinpath(name + '_' + self.stem)
        else:
            return self.parent.joinpath(name + '_' + self.name)

    def append(self, name='', suffix=None):
        """Add extra text after file name, and optionally add extra suffix.
        e.g: blah\blah.extenion ==> becomes ==> blah/blah_name.extension
        """
        if suffix is None:
            suffix = ''.join(self.suffixes)
        return self.parent.joinpath(self.stem + '_' + name + suffix)

    def delete(self, are_you_sure=False):
        if are_you_sure:
            if self.is_file():
                self.unlink()  # missing_ok=True added in 3.8
            else:
                import shutil
                shutil.rmtree(self, ignore_errors=True)
                # self.rmdir()  # dir must be empty
        else:
            print("File not deleted because user is not sure.")

    def send2trash(self):
        import send2trash
        send2trash.send2trash(self.string)

    def move(self, new_path):
        temp = self.absolute()
        temp.rename(new_path.absolute() / temp.name)
        return self

    def renameit(self, new_name):
        self.rename(self.parent / new_name)
        return self

    def copy(self, target=None, verbose=False):
        import shutil
        # assert self.is_file()
        if target is None:
            target = self.append(f"_copy_{get_time_stamp()}")
        shutil.copy(str(self), str(target))  # str() only there for Python < (3.6)
        if verbose:
            print(f"File \n{self}\ncopied successfully to: \n{target}")
        return target

    def clean(self):
        """removes contents on a folder, rather than deleting the folder."""
        self.send2trash()
        self.mkdir()
        return self

    def create(self, parents=True, exist_ok=True, parent_only=False):
        """Creates directory while returning the same object
        """
        if parent_only:
            self.parent.mkdir(parents=parents, exist_ok=exist_ok)
        else:
            self.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    @property
    def browse(self):
        return browse(self)

    def relativity_transform(self, reference='deephead', abs_reference=None):
        """Takes in a path defined relative to reference, transform it to a path relative to execution
        directory, then makes it absolute path.

        .. warning:: reference must be included in the execution directory. Otherwise, absolute path of reference
           should be provided.
        """
        # step one: find absolute path for reference, if not given.
        paths = [P.cwd()] + list(P.cwd().parents)
        names = list(reversed(P.cwd().parts))
        if abs_reference is None:  # find it for reference.
            abs_reference = paths[names.index(reference)]
        return abs_reference / self
        # rel_deephead = os.path.relpath(abs_reference, start=os.curdir)
        # rel_path = os.path.join(rel_deephead, self)
        # abs_path = os.path.abspath(rel_path)
        # return Path(abs_path)

    def split(self, at=None, index=None):
        """Splits a path at a given string or index
        :param self:
        :param at:
        :param index:
        :return: two paths
        """
        if index is None:
            idx = self.parts.index(at)
            return self.split(index=idx)
        else:
            one = self[:index]
            two = P(*self.parts[index:])
            return one, two

    def __getitem__(self, slici):
        return P(*self.parts[slici])

    def myglob(self, pattern='*', r=False, list_=True, files=True, folders=True, dotfiles=False,
               return_type=None,
               absolute=True, filters=None, win_order=False):
        """
        :param win_order:
        :param self:
        :param filters:
        :param dotfiles:
        :param pattern:  regex expression.
        :param r: recursive search
        :param list_: output format, list or generator.
        :param files: include files in search.
        :param folders: include directories in search.
        :param return_type: output type, Pathlib objects or strings.
        :param absolute: return relative paths or abosolute ones.
        :return: search results.

        # :param visible: exclude hidden files and folders (Windows)
        """
        if return_type is None:
            return_type = P

        if filters is None:
            filters = []

        if dotfiles:
            raw = self.glob(pattern) if not r else self.rglob(pattern)
            raw = list(raw)
        else:
            if r:
                path = self / "**" / pattern
                raw = [Path(item) for item in glob(str(path), recursive=r)]
            else:
                path = self.joinpath(pattern)
                raw = [Path(item) for item in glob(str(path))]

        # if os.name == 'nt':
        #     import win32api, win32con

        # def folder_is_hidden(p):
        #     if os.name == 'nt':
        #         attribute = win32api.GetFileAttributes(p)
        #         return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)

        if not raw:  # if empty, don't proceeed
            return raw

        if absolute:
            if not raw[0].is_absolute():
                raw = [item.absolute() for item in raw]

        def run_filter(item):
            flags = [True]
            if not files:
                flags.append(item.is_dir())
            if not folders:
                flags.append(item.is_file())
            for afilter in filters:
                flags.append(afilter(item))
            return all(flags)

        if list_:
            processed = list(filter(run_filter, raw))
            processed = [return_type(item) for item in processed]
            if win_order:
                processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', x.stem)])
            return List(processed)
        else:
            def generator():
                flag = False
                while not flag:
                    item = next(raw)
                    flag = run_filter(item)
                    if flag:
                        yield return_type(item)

            return generator

    def listdir(self):
        contents = self.glob('*')
        return List([apath.stem for apath in contents])

    def find(self, *args, **kwargs):
        """short for globbing then using next method to get the first result
        """
        results = self.myglob(*args, **kwargs)
        return results[0] if len(results) > 0 else None

    def readit(self, reader=None, **kwargs):
        if reader is None:
            reader = getattr(Read, self.suffix[1:])
        return reader(str(self), **kwargs)

    def explore(self):
        os.startfile(os.path.realpath(self))

    def __repr__(self):  # this is useful only for the console
        return "AlexPath(" + self.__str__() + ")"

    @property
    def string(self):  # this method is used by other functions to get string representation of path
        return str(self)

    @staticmethod
    def tmp(folder=None, fn=None, path="home"):
        """
        folder is created.
        file name is not created, only appended.
        """
        if str(path) == "home":
            path = P.home() / f"tmp_results"
            path.mkdir(exist_ok=True, parents=True)
        if folder is not None:
            path = path / folder
            path.mkdir(exist_ok=True, parents=True)
        if fn is not None:
            path = path / fn
        return path

        # funcs = list(filter(lambda x: '__' not in x, list(MyPath.__dict__)))
        # [setattr(Path, name, getattr(MyPath, name)) for name in funcs]


tmp = P.tmp


def browse(path, depth=8, width=20):
    """
    :param width: if there are more than this items in a directory, dont' parse the rest.
    :param depth: to prevent crash, limit how deep recursive call can happen.
    :param path: absolute path
    :return: constructs a class dynamically by using object method.
    """
    if depth > 0:
        my_dict = {'z_path': P(path)}  # prepare _path attribute which returns current path from the browser object
        val_paths = glob(os.path.join(path, '*'))  # prepare other methods that refer to the contents.
        temp = [os.path.basename(i) for i in val_paths]
        # this is better than listdir, gives consistent results with glob (no hidden files)
        key_contents = []  # keys cannot be folders/file names immediately, there are caveats.
        for akey in temp:
            # if not akey[0].isalpha():  # cannot start with digit or +-/?.,<>{}\|/[]()*&^%$#@!~`
            #     akey = '_' + akey
            for i in string.punctuation.replace('_', ' '):  # disallow punctuation and space except for _
                akey = akey.replace(i, '_')
            key_contents.append(akey)  # now we have valid attribute name
        for i, (akey, avalue) in enumerate(zip(key_contents, val_paths)):
            if i < width:
                if os.path.isfile(avalue):
                    my_dict['FLE_' + akey] = P(avalue)
                else:
                    my_dict['FDR_' + akey] = browse(avalue, depth=depth - 1)

        def repr_func(self):
            if self.z_path.is_file():
                return 'Explorer object. File: \n' + str(self.z_path)
            else:
                return 'Explorer object. Folder: \n' + str(self.z_path)

        def str_func(self):
            return str(self.z_path)

        my_dict["__repr__"] = repr_func
        my_dict["__str__"] = str_func
        my_class = type(os.path.basename(path), (), dict(zip(my_dict.keys(), my_dict.values())))
        return my_class()
    else:
        return path


class Read:
    @staticmethod
    def read(path):
        return P(path).readit()

    @staticmethod
    def npy(path):
        data = np.load(path, allow_pickle=True)
        if data.dtype == np.object:
            data = data.all()
        return data

    @staticmethod
    def pickle(path):
        import pickle
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj

    @staticmethod
    def nii(path):
        import nibabel as nib
        return nib.load(path)

    # @staticmethod
    # def dicom(directory):
    #     import dicom2nifti
    #     import dicom2nifti.settings as settings
    #     settings.disable_validate_orthogonal()
    #     settings.enable_resampling()
    #     settings.set_resample_spline_interpolation_order(1)
    #     settings.set_resample_padding(-1000)
    #     dicom2nifti.convert_directory(directory, directory)
    #     return Path(directory).glob('*.nii').__next__()

    @staticmethod
    def gz(fn):
        import shutil
        import gzip
        fn = Path(fn)
        name = fn.parent / fn.stem
        with gzip.open(fn, 'r') as f_in, open(name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return name

    @staticmethod
    def tar(fn, mode='r', fname=None, **kwargs):
        import tarfile
        file = tarfile.open(fn, mode)
        if fname is None:  # extract all files in the archive
            file.extractall(**kwargs)
        else:
            file.extract(fname, **kwargs)
        file.close()

    @staticmethod
    def zip(fn, op_path=None):
        from zipfile import ZipFile
        if op_path is None:
            op_path = fn.parent / fn.stem
        with ZipFile(fn, 'r') as zipObj:
            zipObj.extractall(op_path)
        return op_path

    @staticmethod
    def mat(path, correct_dims=True, single_item=False):
        """
        :param path:
        :param correct_dims:
        :param single_item: if the .mat file contains a single item, not need to return a dictionary.
                            the default is False cause it is not standard behaviour and possibly confusing.
        :return:
        """
        try:  # try the old version
            from scipy.io import loadmat
            data = loadmat(path)
            keys = list(data.keys())
            for akey in ['__header__', '__version__', '__globals__']:
                keys.remove(akey)
            data['_data'] = [data[i] for i in keys]
            data['_names'] = keys
            if len(data.keys()) == 3 + 2 + 1 and single_item:  # single data item
                return data['_data'][0]
        except NotImplementedError:
            import h5py  # For Matlab v7.3 files, we need:
            f = h5py.File(path, mode='r')  # returns an object
            data = {}
            for item in f:
                temp = np.array(f[item], order='F')  # Now you get the correct shape.
                if correct_dims:
                    n = len(temp.shape)
                    arrangements = tuple(range(n - 2)) + (n - 1, n - 2)
                    temp = temp.transpose(arrangements)
                data[item] = temp
            f.close()
        return data

    @staticmethod
    def json(path):
        import json
        with open(str(path), "r") as file:
            return json.load(file)


class Save:
    @staticmethod
    def mat(path=P('.'), mdict=None):
        from scipy.io import savemat
        if '.mat' not in str(path):
            path += '.mat'
        path.parent.mkdir(exist_ok=True, parents=True)
        for key, value in mdict.items():
            if value is None:
                mdict[key] = []
        savemat(str(path), mdict)

    @staticmethod
    def json(path, obj):
        import json
        if not str(path).endswith(".json"):
            path = str(path) + ".json"
        with open(str(path), "w") as file:
            json.dump(obj, file)

    @staticmethod
    def pickle(path, obj):
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def compress_nii(path, delete=True):
        # compress nii with gz
        files = P(path).myglob('*.nii')
        import gzip
        import shutil
        for file in tqdm(files):
            with open(file, 'rb') as f_in:
                with gzip.open(str(file) + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        if delete:
            for afile in files:
                afile.unlink()


def accelerate(func, ip):
    """ Conditions for this to work:
    * Must run under __main__ context
    * func must be defined outside that context.


    To accelerate IO-bound process, use multithreading. An example of that is somthing very cheap to process,
    but takes a long time to be obtained like a request from server. For this, multithreading launches all threads
    together, then process them in an interleaved fashion as they arrive, all will line-up for same processor,
    if it happens that they arrived quickly.

    To accelerate processing-bound process use multiprocessing, even better, use Numba.
    Method1 use: multiprocessing / multithreading.
    Method2: using joblib (still based on multiprocessing)
    from joblib import Parallel, delayed
    Fast method using Concurrent module
    """
    split = np.array_split(ip, os.cpu_count())
    # make each thread process multiple inputs to avoid having obscene number of threads with simple fast
    # operations

    # vectorize the function so that it now accepts lists of ips.
    # def my_func(ip):
    #     return [func(tmp) for tmp in ip]

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        op = executor.map(func, split)
        op = list(op)  # convert generator to list
    op = np.concatenate(op, axis=0)
    # op = self.reader.assign_resize(op, f=0.8, nrp=56, ncp=47, interpolation=True)
    return op


class Manipulator:
    @staticmethod
    def reverse(my_map):
        return {v: k for k, v in my_map.items()}

    @staticmethod
    def concat_dicts(*dicts, lenient=True):
        if not lenient:
            keys = dicts[0].keys()
            for i in dicts[1:]:
                assert i.keys() == keys
        total_dict = dicts[0].copy()  # take first dict in the tuple
        if len(dicts) > 1:  # are there more tuples?
            for key in total_dict.keys():
                for adict in dicts[1:]:
                    try:
                        total_dict[key] += adict[key]
                    except KeyError:
                        pass
        return total_dict

    @staticmethod
    def merge_adjacent_axes(array, ax1, ax2):
        """Multiplies out two axes to generate reduced order array.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        """
        shape = array.shape
        # order = len(shape)
        sz1, sz2 = shape[ax1], shape[ax2]
        new_shape = shape[:ax1] + (sz1 * sz2,) + shape[ax2 + 1:]
        return array.reshape(new_shape)

    @staticmethod
    def merge_axes(array, ax1, ax2):
        """Brings ax2 next to ax1 first, then combine the two axes into one.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        """
        array2 = np.moveaxis(array, ax2, ax1 + 1)  # now, previously known as ax2 is located @ ax1 + 1
        return Manipulator.merge_adjacent_axes(array2, ax1, ax1 + 1)

    @staticmethod
    def expand_axis(array, ax_idx, factor):
        total_shape = list(array.shape)
        size = total_shape.pop(ax_idx)
        new_shape = (int(size / factor), factor)
        for index, item in enumerate(new_shape):
            total_shape.insert(ax_idx + index, item)
        return array.reshape(tuple(total_shape))

    @staticmethod
    def slicer(array, lower_, upper_):
        n = len(array)
        lower_ = lower_ % n  # if negative, you get the positive equivalent. If > n, you get principal value.
        roll = lower_
        lower_ = lower_ - roll
        upper_ = upper_ - roll
        array_ = np.roll(array, -roll, axis=0)
        upper_ = upper_ % n
        return array_[lower_: upper_]


def batcher(func_type='function'):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, x, *args, per_instance_kwargs=None, **kwargs):
                output = []
                for counter, item in enumerate(x):
                    if per_instance_kwargs is not None:
                        mykwargs = {key: value[counter] for key, value in per_instance_kwargs.items()}
                    else:
                        mykwargs = {}
                    output.append(func(self, item, *args, **mykwargs, **kwargs))
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __init__(self, func):
                self.func = func

            def __call__(self, x, **kwargs):
                output = [self.func(item, **kwargs) for item in x]
                return np.array(output)

        return Batch


def batcherv2(func_type='function', order=1):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, *args, **kwargs):
                output = [func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __int__(self, func):
                self.func = func

            def __call__(self, *args, **kwargs):
                output = [self.func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

        return Batch


class DisplayData:
    def __init__(self, x):
        self.x = pd.DataFrame(x)

    @staticmethod
    def set_display():
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 200)
        pd.set_option('display.max_colwidth', 40)

    @staticmethod
    def eng():
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        pd.options.display.float_format = '{:, .5f}'.format
        pd.set_option('precision', 7)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class List(list):
    def __init__(self, obj_list=None):
        super().__init__()
        self.list = obj_list if obj_list is not None else []
        # if len(self.list) > 0:
        #     class Fake(self.list[0].__class__):
        #         def __getattr__(self, item):
        #             pass
        #     self.example = Fake

    @classmethod
    def replica(cls, obj, count):
        from copy import deepcopy
        return cls([deepcopy(obj) for _ in range(count)])

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
            _ = memodict
        return List([copy.deepcopy(i) for i in self.list])

    def __copy__(self):
        return self.__deepcopy__()

    def method(self, name, *args, **kwargs):
        return List([getattr(i, name)(*args, **kwargs) for i in self.list])

    def attr(self, name):
        return List([getattr(i, name) for i in self.list])

    def __getattr__(self, name):
        result = List([getattr(i, name) for i in self.list])
        return result

    def __call__(self, *args, lest=True, **kwargs):
        if lest:
            return List([i(*args, **kwargs) for i in self.list])
        else:
            return [i(*args, **kwargs) for i in self.list]

    def __getitem__(self, key):
        result = self.list[key]
        if type(key) is not slice:
            return result  # choose one item
        else:
            return List(result)

    def find(self, string_):
        """Looks up the string representation of all items in the list and finds the one that partially matches
        the argument passed. This method is a short for self.filter(lambda x: string_ in str(x)) If you need more
        complicated logic in the search, revert to filter method.
        """
        for item in self.list:
            if string_ in str(item):
                return item
        return None

    def append(self, obj):
        self.list.append(obj)

    def __add__(self, other):
        return List(self.list + other.list)

    def __repr__(self):
        if len(self.list) > 0:
            tmp1 = f"List object with {len(self.list)}"
            tmp2 = f" elements. One example of those elements: \n{self.list[0].__repr__()}"
            return tmp1 + tmp2
        else:
            return f"An Empty List []"

    def __len__(self):
        return len(self.list)

    @property
    def len(self):
        return self.list.__len__()

    def __iter__(self):
        return iter(self.list)

    @property
    def np(self):
        return np.array(self.list)

    def combine(self):
        res = self.list[0]
        for item in self.list[1:]:
            res = res + item
        return res

    def apply(self, func, *args, lest=None, jobs=None, depth=1, **kwargs):
        """
        :param jobs:
        :param func: func has to be a function, possibly a lambda function. At any rate, it should return something.
        :param args:
        :param lest:
        :param depth: apply the function to inner Lists
        :param kwargs: a list of outputs each time the function is called on elements of the list.
        :return:
        """
        if depth > 1:
            depth -= 1
            # assert type(self.list[0]) == List, "items are not Lists".
            self.apply(lambda x: x.apply(func, *args, lest=lest, jobs=jobs, depth=depth, **kwargs))

        if type(func) is str:
            func = eval("lambda x: " + func)

        from joblib import Parallel, delayed
        if lest is None:
            if jobs:
                return List(Parallel(n_jobs=jobs)(delayed(func)(i, *args, **kwargs) for i in tqdm(self.list)))
            else:
                return List([func(i, *args, **kwargs) for i in self.list])
        else:
            if jobs:
                return List(Parallel(n_jobs=jobs)(delayed(func)(obj, obj_) for obj, obj_ in tqdm(zip(self.list, lest))))
            else:
                return List([func(obj, obj_) for obj, obj_ in zip(self.list, lest)])

    def modify(self, func, lest=None):
        """Modifies objects rather than returning new list of objects, hence the name of the method.
        :param func: a string that will be executed, assuming idx, x and y are given.
        :param lest:
        :return:
        """
        if lest is None:
            for x in self.list:
                _ = x
                exec(func)
        else:
            for idx, (x, y) in enumerate(zip(self.list, lest)):
                _, _, _ = idx, x, y
                exec(func)
        return self

    def sort(self, *args, **kwargs):
        self.list.sort(*args, **kwargs)

    def sorted(self, *args, **kwargs):
        return List(sorted(self.list, *args, **kwargs))

    def idx(self, start, end=None, step=None):
        """Used to access entries of items
        """
        return List([item[start:end:step] for item in self.list])

    def filter(self, func):
        if type(func) is str:
            func = eval("lambda x: " + func)
        result = List()
        for item in self.list:
            if func(item):
                result.append(item)
        return result

    def print(self, nl=1, sep=False, char='-'):
        for idx, item in enumerate(self.list):
            print(f"{idx:2}- {item}", end=' ')
            for _ in range(nl):
                print('', end='\n')
            if sep:
                print(char*100)

    def df(self):
        DisplayData.set_display()
        columns = ['object'] + list(self.list[0].__dict__.keys())
        df = pd.DataFrame(columns=columns)
        for i, obj in enumerate(self.list):
            df.loc[i] = [obj] + list(self.list[i].__dict__.values())
        return df


def assert_package_installed(package):
    try:
        __import__(package)
    except ImportError:
        import pip
        pip.main(['install', package])


def run(name, asis=False):
    import inspect
    import textwrap
    assert_package_installed("clipboard")
    import clipboard

    codelines = inspect.getsource(name)
    if not asis:
        # remove def func_name() line from the list, and return statement
        idx = codelines.find("):\n")
        codelines = codelines[idx + 4:]

        # remove any indentation (4 for funcs and 8 for classes methods, etc)
        codelines = textwrap.dedent(codelines)

        # remove return statements
        codelines = codelines.split("\n")
        codelines = [code + "\n" for code in codelines if not code.startswith("return ")]

    code_string = ''.join(codelines)  # convert list to string.

    temp = inspect.getfullargspec(name)
    arg_string = """"""
    # if isinstance(type(name), types.MethodType) else tmp.args
    for key, val in zip(temp.args[1:], temp.defaults):
        arg_string += f"{key} = {val}\n"

    result = arg_string + code_string
    clipboard.copy(result)
    return result  # ready to be run with exec()


class Log:
    def __init__(self, path=None):
        if path is None:
            path = P('console_output')
        self.path = path + '.log'
        sys.stdout = open(self.path, 'w')

    def finish(self):
        sys.stdout.close()
        print(f"Finished ... have a look @ \n {self.path}")


def classisize(dictionary, name=None):
    class Classizer:
        pass
    obj = Classizer()  # we have to add attributes to instance not to class to avoid sharing.
    obj.__dict__ = dictionary
    obj.name = name
    return obj


def polygon_area(points):
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
